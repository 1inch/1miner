//! The four address derivations the miner searches over.
//!
//! All of them end in "keccak something, take the low 20 bytes". What differs
//! is the pre-image, and getting a pre-image subtly wrong yields a plausible
//! address that simply does not exist on chain, so each derivation below is
//! covered by a known-answer test.

use crate::keccak256;

pub type Address = [u8; 20];
pub type Hash = [u8; 32];
pub type Salt = [u8; 32];

/// Minimal CREATE3 proxy (Solady/solmate). Deployed via CREATE2, it then
/// CREATEs the real contract, which is why a CREATE3 address does not depend
/// on the deployed contract's init code.
pub const PROXY_CHILD_BYTECODE: [u8; 16] =
    [0x67, 0x36, 0x3d, 0x3d, 0x37, 0x36, 0x3d, 0x34, 0xf0, 0x3d, 0x52, 0x60, 0x08, 0x60, 0x18, 0xf3];

/// `keccak256(PROXY_CHILD_BYTECODE)`, asserted in tests rather than trusted.
pub const DEFAULT_PROXY_CODE_HASH: Hash = [
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f,
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f,
];

/// The 1inch Address NFT deployer. Provided for documentation and tests only;
/// the CLI never substitutes it, because mining a magic against an assumed
/// deployer produces a result that looks valid and is unusable.
pub const ONEINCH_NFT_DEPLOYER: Address = [
    0x1a, 0xdd, 0x4e, 0x55, 0xec, 0xef, 0xfd, 0x79, 0x5b, 0x01, 0xd2, 0x22, 0x03, 0xd2, 0x80, 0xc9,
    0x3a, 0x2f, 0x1d, 0xc3,
];

fn low20(hash: Hash) -> Address {
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..32]);
    addr
}

/// Externally owned account address: `keccak256(pubkey_x || pubkey_y)[12:]`,
/// where the key is uncompressed and the `0x04` prefix is omitted.
pub fn eoa_address(pubkey_x: &[u8; 32], pubkey_y: &[u8; 32]) -> Address {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(pubkey_x);
    buf[32..].copy_from_slice(pubkey_y);
    low20(keccak256(&buf))
}

/// Legacy CREATE: `keccak256(rlp([sender, nonce]))[12:]`.
///
/// Only nonces below 0x80 are supported, which covers both callers here:
/// nonce 0 for profanity's `--contract` scoring and nonce 1 for the CREATE3
/// proxy. RLP encodes a zero nonce as the empty string `0x80`, not as `0x00`.
pub fn create_address(sender: &Address, nonce: u8) -> Address {
    assert!(nonce < 0x80, "create_address supports single-byte nonces below 0x80");
    let mut buf = [0u8; 23];
    buf[0] = 0xd6; // list, 22 bytes payload
    buf[1] = 0x94; // string, 20 bytes
    buf[2..22].copy_from_slice(sender);
    buf[22] = if nonce == 0 { 0x80 } else { nonce };
    low20(keccak256(&buf))
}

/// EIP-1014 CREATE2: `keccak256(0xff || deployer || salt || init_code_hash)[12:]`.
pub fn create2_address(deployer: &Address, salt: &Salt, init_code_hash: &Hash) -> Address {
    low20(keccak256(&create2_preimage(deployer, salt, init_code_hash)))
}

/// The 85-byte CREATE2 pre-image. Exposed because the GPU kernels bake this
/// exact buffer in as a constant and mutate three words of the salt in place.
pub fn create2_preimage(deployer: &Address, salt: &Salt, init_code_hash: &Hash) -> [u8; 85] {
    let mut buf = [0u8; 85];
    buf[0] = 0xff;
    buf[1..21].copy_from_slice(deployer);
    buf[21..53].copy_from_slice(salt);
    buf[53..85].copy_from_slice(init_code_hash);
    buf
}

/// CREATE3: CREATE2 a proxy, then the proxy CREATEs at nonce 1. The result
/// depends only on the factory and the salt.
pub fn create3_address(factory: &Address, salt: &Salt, proxy_code_hash: &Hash) -> Address {
    let proxy = create2_address(factory, salt, proxy_code_hash);
    create_address(&proxy, 1)
}

/// 1inch Address NFT salt layout: the mined 16-byte magic in the high half,
/// the account the vanity address is minted for pinned into the low half.
pub fn nft_salt(magic: &[u8; 16], mint_for: &Address) -> Salt {
    let digest = keccak256(mint_for);
    let mut salt = [0u8; 32];
    salt[..16].copy_from_slice(magic);
    salt[16..].copy_from_slice(&digest[16..32]);
    salt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexutil::parse_address;

    #[test]
    fn proxy_code_hash_matches_bytecode() {
        assert_eq!(keccak256(&PROXY_CHILD_BYTECODE), DEFAULT_PROXY_CODE_HASH);
    }

    /// Cross-checked three ways upstream: the Rust miner unit test, the Foundry
    /// test in Create3.t.sol, and `cast keccak` by hand.
    #[test]
    fn create3_known_vector() {
        let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let expected = parse_address("0x6c8ed9dc3734d7944beddd2fb5acdf5f17247870").unwrap();
        assert_eq!(
            create3_address(&factory, &[0u8; 32], &DEFAULT_PROXY_CODE_HASH),
            expected
        );
    }

    /// The intermediate proxy is worth pinning separately: if only the final
    /// address were checked, a compensating error in both steps could hide.
    #[test]
    fn create3_intermediate_proxy() {
        let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let proxy = create2_address(&factory, &[0u8; 32], &DEFAULT_PROXY_CODE_HASH);
        assert_eq!(
            proxy,
            parse_address("0x932A2198eC22043b9702a6250C8Ad906a3D62131").unwrap()
        );
    }

    /// Official EIP-1014 examples.
    #[test]
    fn create2_eip1014_vectors() {
        let cases: [(&str, &str, &str, &str); 6] = [
            (
                "0x0000000000000000000000000000000000000000",
                "0x0000000000000000000000000000000000000000000000000000000000000000",
                "0x00",
                "0x4D1A2e2bB4F88F0250f26Ffff098B0b30B26BF38",
            ),
            (
                "0xdeadbeef00000000000000000000000000000000",
                "0x0000000000000000000000000000000000000000000000000000000000000000",
                "0x00",
                "0xB928f69Bb1D91Cd65274e3c79d8986362984fDA3",
            ),
            (
                "0xdeadbeef00000000000000000000000000000000",
                "0x000000000000000000000000feed000000000000000000000000000000000000",
                "0x00",
                "0xD04116cDd17beBE565EB2422F2497E06cC1C9833",
            ),
            (
                "0x0000000000000000000000000000000000000000",
                "0x0000000000000000000000000000000000000000000000000000000000000000",
                "0xdeadbeef",
                "0x70f2b2914A2a4b783FaEFb75f459A580616Fcb5e",
            ),
            (
                "0x00000000000000000000000000000000deadbeef",
                "0x00000000000000000000000000000000000000000000000000000000cafebabe",
                "0xdeadbeef",
                "0x60f3f640a8508fC6a86d45DF051962668E1e8AC7",
            ),
            (
                "0x0000000000000000000000000000000000000000",
                "0x0000000000000000000000000000000000000000000000000000000000000000",
                "0x",
                "0xE33C0C7F7df4809055C3ebA6c09CFe4BaF1BD9e0",
            ),
        ];

        for (deployer, salt, init_code, expected) in cases {
            let deployer = parse_address(deployer).unwrap();
            let salt = crate::hexutil::parse_hash(salt).unwrap();
            let init_code = crate::hexutil::parse_hex(init_code).unwrap();
            let expected = parse_address(expected).unwrap();
            assert_eq!(
                create2_address(&deployer, &salt, &keccak256(&init_code)),
                expected,
                "deployer {}",
                hex::encode(deployer)
            );
        }
    }

    /// A zero nonce is RLP `0x80`, not `0x00`; getting this wrong silently
    /// shifts every profanity `--contract` result.
    #[test]
    fn create_nonce_encoding() {
        let sender = parse_address("0x6ac7ea33f8831ea9dcc53393aaa88b25a785dbf0").unwrap();
        assert_eq!(
            create_address(&sender, 0),
            parse_address("0xcd234a471b72ba2f1ccf0a70fcaba648a5eecd8d").unwrap()
        );
        assert_eq!(
            create_address(&sender, 1),
            parse_address("0x343c43a37d37dff08ae8c4a11544c718abb4fcf8").unwrap()
        );
        assert_ne!(create_address(&sender, 0), create_address(&sender, 1));
    }

    /// Known-answer vectors taken from the 1inch AddressToken contract's own
    /// test suite, which deploys the contract at a fixed address and asserts
    /// these magic-to-address pairs.
    ///
    /// This is the only external check on the 1nft derivation, and it pins
    /// several things at once: the salt layout (magic in the high 16 bytes,
    /// `keccak256(account)` masked into the low 16), the standard proxy code
    /// hash, and the fact that solmate's CREATE3 uses the calling contract as
    /// the CREATE2 deployer rather than `msg.sender`. If any of those were
    /// wrong, none of these would match.
    #[test]
    fn nft_vectors_from_the_address_token_test_suite() {
        // AddressToken as deployed by its own hardhat fixture.
        let deployer = parse_address("0x5FbDB2315678afecb367f032d93F642f64180aa3").unwrap();
        // The first hardhat signer, which mints for itself in those tests.
        let account = parse_address("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266").unwrap();

        let cases = [
            ("83e07be8812a93bc76504bc8c10f79c7", "0x000000a6C09bd7f6Ba10642DBaCe1bE60565A2F8"),
            ("a245d3d1f4bc5e70beb85db24b6f4df1", "0xc07EC7da97D7444F738e955f28F6C91d15000000"),
            ("4a7beee747938760a0fbd441e3ce93f5", "0x6828985368578258349260531646495303581057"),
            ("057b69fd8b880100129d0f0000000000", "0x6666665e4d6a736100A7D8eD5dfBacDf99f29DFf"),
            ("00000000000000000000000000000000", "0x89E802345bfB6CaD865fb5935fb6749D65D25764"),
        ];

        for (magic_hex, expected) in cases {
            let magic = crate::hexutil::parse_magic(magic_hex).unwrap();
            let salt = nft_salt(&magic, &account);
            let got = create3_address(&deployer, &salt, &DEFAULT_PROXY_CODE_HASH);
            assert_eq!(
                got,
                parse_address(expected).unwrap(),
                "magic {magic_hex} should mint {expected}"
            );
        }
    }

    #[test]
    fn nft_salt_pins_low_half_to_caller() {        let caller = parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        let magic = [0xABu8; 16];
        let salt = nft_salt(&magic, &caller);
        assert_eq!(&salt[..16], &magic);
        assert_eq!(&salt[16..], &keccak256(&caller)[16..32]);
    }

    #[test]
    fn create2_preimage_layout() {
        let deployer = [0x11u8; 20];
        let salt = [0x22u8; 32];
        let code_hash = [0x33u8; 32];
        let buf = create2_preimage(&deployer, &salt, &code_hash);
        assert_eq!(buf[0], 0xff);
        assert_eq!(&buf[1..21], &deployer);
        assert_eq!(&buf[21..53], &salt);
        assert_eq!(&buf[53..85], &code_hash);
    }
}
