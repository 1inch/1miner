//! Mining modes and the kernel constants they resolve to.
//!
//! The three salt modes share one kernel. They differ only in what goes into
//! the 85-byte CREATE2 pre-image, whether a second keccak runs afterwards, and
//! whether part of the salt is pinned.

use crate::{
    Address, CoreError, Hash, Result, Salt,
    address::{create_address, create2_address, create2_preimage, create3_address},
    keccak256,
    secp256k1::Point,
};

/// The keccak state the kernels operate on: 200 bytes, addressable as 50
/// little-endian u32 words. Only three of those words vary per work-item.
pub const STATE_BYTES: usize = 200;
pub const STATE_WORDS: usize = 50;

/// Word indices the kernel bumps, from eradicate2.cl / eradicate3.cl:
/// `h.d[6] += deviceIndex; h.d[7] += get_global_id(0); h.d[8] += round;`
pub const WORD_DEVICE: usize = 6;
pub const WORD_GLOBAL_ID: usize = 7;
pub const WORD_ROUND: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MineMode {
    Profanity,
    Create2,
    Create3,
    /// 1inch Address NFT. Spelled `1nft` on the command line.
    Nft,
}

impl MineMode {
    pub fn as_str(self) -> &'static str {
        match self {
            MineMode::Profanity => "profanity",
            MineMode::Create2 => "create2",
            MineMode::Create3 => "create3",
            MineMode::Nft => "1nft",
        }
    }

    pub fn is_salt_mode(self) -> bool {
        !matches!(self, MineMode::Profanity)
    }

    /// CREATE3 and 1nft run the extra CREATE keccak; CREATE2 stops after one.
    pub fn needs_second_hash(self) -> bool {
        matches!(self, MineMode::Create3 | MineMode::Nft)
    }
}

/// Everything the salt kernel needs baked in.
#[derive(Debug, Clone)]
pub struct SaltConfig {
    pub mode: MineMode,
    pub deployer: Address,
    /// `keccak256(initCode)` for CREATE2, the proxy bytecode hash for CREATE3.
    pub code_hash: Hash,
    /// Randomised per run. For 1nft the low 16 bytes are pinned to the account.
    pub base_salt: Salt,
    /// 1nft only: the account the address is minted for.
    pub mint_for: Option<Address>,
}

impl SaltConfig {
    pub fn new(
        mode: MineMode,
        deployer: Address,
        code_hash: Hash,
        base_salt: Salt,
        mint_for: Option<Address>,
    ) -> Result<Self> {
        if !mode.is_salt_mode() {
            return Err(CoreError::Config(format!(
                "{} is not a salt-searching mode",
                mode.as_str()
            )));
        }
        match (mode, mint_for) {
            (MineMode::Nft, None) => {
                return Err(CoreError::Config(
                    "1nft requires --mint-for, the account the address is minted for".into(),
                ));
            }
            (MineMode::Create2 | MineMode::Create3, Some(_)) => {
                return Err(CoreError::Config(format!(
                    "--mint-for only applies to 1nft, not {}",
                    mode.as_str()
                )));
            }
            _ => {}
        }

        let mut cfg = Self {
            mode,
            deployer,
            code_hash,
            base_salt,
            mint_for,
        };
        if let Some(account) = cfg.mint_for {
            // Pin the low half so the deployer's own salt derivation matches.
            let digest = keccak256(&account);
            cfg.base_salt[16..].copy_from_slice(&digest[16..32]);
        }
        Ok(cfg)
    }

    /// The 200-byte keccak state with the CREATE2 pre-image and pad bit in
    /// place. The trailing `0x80` pad byte is *not* here: the kernels fold it
    /// into the permutation itself, and the CPU reference does the same.
    pub fn state(&self) -> [u8; STATE_BYTES] {
        let preimage = create2_preimage(&self.deployer, &self.base_salt, &self.code_hash);
        let mut state = [0u8; STATE_BYTES];
        state[..85].copy_from_slice(&preimage);
        state[85] ^= 0x01;
        state
    }

    /// The baked-in constant handed to the kernel as 25 little-endian u64s.
    pub fn state_words(&self) -> [u64; 25] {
        let state = self.state();
        let mut words = [0u64; 25];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u64::from_le_bytes(state[i * 8..i * 8 + 8].try_into().unwrap());
        }
        words
    }

    /// Reproduce the salt a given work-item used.
    ///
    /// This mirrors the reconstruction inside `..._result_update`, which is
    /// separate code from the hashing path in the kernel. Recomputing it here
    /// is what catches the two drifting apart.
    pub fn salt_at(&self, device_index: u32, global_id: u32, round: u32) -> Salt {
        let state = self.state();
        let mut words = [0u32; STATE_WORDS];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes(state[i * 4..i * 4 + 4].try_into().unwrap());
        }
        words[WORD_DEVICE] = words[WORD_DEVICE].wrapping_add(device_index);
        words[WORD_GLOBAL_ID] = words[WORD_GLOBAL_ID].wrapping_add(global_id);
        words[WORD_ROUND] = words[WORD_ROUND].wrapping_add(round);

        let mut bytes = [0u8; STATE_BYTES];
        for (i, word) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut salt = [0u8; 32];
        salt.copy_from_slice(&bytes[21..53]);
        salt
    }

    /// Derive the address a salt produces under this mode.
    pub fn address_for_salt(&self, salt: &Salt) -> Address {
        if self.mode.needs_second_hash() {
            create3_address(&self.deployer, salt, &self.code_hash)
        } else {
            create2_address(&self.deployer, salt, &self.code_hash)
        }
    }

    /// Convenience for the round-trip check: work-item coordinates to address.
    pub fn address_at(&self, device_index: u32, global_id: u32, round: u32) -> Address {
        self.address_for_salt(&self.salt_at(device_index, global_id, round))
    }

    /// For 1nft the reportable result is the high 16 bytes of the salt.
    pub fn magic_at(&self, device_index: u32, global_id: u32, round: u32) -> Option<[u8; 16]> {
        if self.mode != MineMode::Nft {
            return None;
        }
        let salt = self.salt_at(device_index, global_id, round);
        let mut magic = [0u8; 16];
        magic.copy_from_slice(&salt[..16]);
        Some(magic)
    }
}

/// Profanity mode inputs. Only a public key is ever accepted, so the miner
/// cannot learn a private key even when run on rented hardware.
#[derive(Debug, Clone)]
pub struct ProfanityConfig {
    pub seed_public_key: Point,
    /// Score the contract address this key would deploy at nonce 0 instead of
    /// the account address itself.
    pub contract: bool,
}

impl ProfanityConfig {
    pub fn address_for_point(&self, point: &Point) -> Address {
        let account = point.address();
        if self.contract {
            create_address(&account, 0)
        } else {
            account
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModeConfig {
    Profanity(ProfanityConfig),
    Salt(SaltConfig),
}

impl ModeConfig {
    pub fn mode(&self) -> MineMode {
        match self {
            ModeConfig::Profanity(_) => MineMode::Profanity,
            ModeConfig::Salt(s) => s.mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::DEFAULT_PROXY_CODE_HASH;
    use crate::hexutil::parse_address;

    fn salt_cfg(mode: MineMode, caller: Option<Address>) -> SaltConfig {
        let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let mut base = [0u8; 32];
        for (i, b) in base.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(11).wrapping_add(5);
        }
        SaltConfig::new(mode, deployer, DEFAULT_PROXY_CODE_HASH, base, caller).unwrap()
    }

    #[test]
    fn state_layout_has_preimage_and_pad_bit() {
        let cfg = salt_cfg(MineMode::Create3, None);
        let state = cfg.state();
        assert_eq!(state[0], 0xff);
        assert_eq!(&state[1..21], &cfg.deployer);
        assert_eq!(&state[21..53], &cfg.base_salt);
        assert_eq!(&state[53..85], &cfg.code_hash);
        assert_eq!(state[85], 0x01);
        // The trailing 0x80 belongs to the permutation, not the state.
        assert!(state[86..].iter().all(|b| *b == 0));
    }

    #[test]
    fn zero_work_item_reproduces_the_base_salt() {
        let cfg = salt_cfg(MineMode::Create2, None);
        assert_eq!(cfg.salt_at(0, 0, 0), cfg.base_salt);
    }

    #[test]
    fn work_item_coordinates_change_only_the_three_words() {
        let cfg = salt_cfg(MineMode::Create2, None);
        let base = cfg.salt_at(0, 0, 0);
        let bumped = cfg.salt_at(1, 2, 3);
        // Salt bytes 0..3 (state bytes 21..24) sit below word 6 and never move.
        assert_eq!(&base[..3], &bumped[..3]);
        assert_ne!(base, bumped);
        // Bytes 15.. (state byte 36 onward) are above word 8 and never move.
        assert_eq!(&base[15..], &bumped[15..]);
    }

    #[test]
    fn create2_and_create3_disagree_for_the_same_salt() {
        let two = salt_cfg(MineMode::Create2, None);
        let three = salt_cfg(MineMode::Create3, None);
        let salt = two.salt_at(0, 7, 9);
        assert_ne!(two.address_for_salt(&salt), three.address_for_salt(&salt));
    }

    #[test]
    fn nft_pins_the_low_half_and_exposes_the_magic() {
        let caller = parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        let cfg = salt_cfg(MineMode::Nft, Some(caller));
        let salt = cfg.salt_at(0, 5, 2);
        assert_eq!(&salt[16..], &keccak256(&caller)[16..32]);

        let magic = cfg.magic_at(0, 5, 2).unwrap();
        assert_eq!(&salt[..16], &magic);
        // The magic plus the caller reconstructs the full salt the deployer uses.
        assert_eq!(crate::address::nft_salt(&magic, &caller), salt);
    }

    #[test]
    fn nft_requires_a_caller_and_others_reject_one() {
        let deployer = [0u8; 20];
        assert!(
            SaltConfig::new(
                MineMode::Nft,
                deployer,
                DEFAULT_PROXY_CODE_HASH,
                [0; 32],
                None
            )
            .is_err()
        );
        assert!(
            SaltConfig::new(
                MineMode::Create3,
                deployer,
                DEFAULT_PROXY_CODE_HASH,
                [0; 32],
                Some(deployer)
            )
            .is_err()
        );
        assert!(
            SaltConfig::new(
                MineMode::Profanity,
                deployer,
                DEFAULT_PROXY_CODE_HASH,
                [0; 32],
                None
            )
            .is_err()
        );
    }

    #[test]
    fn state_words_round_trip_to_state_bytes() {
        let cfg = salt_cfg(MineMode::Create3, None);
        let words = cfg.state_words();
        let mut bytes = [0u8; STATE_BYTES];
        for (i, w) in words.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(bytes, cfg.state());
    }
}
