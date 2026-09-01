//! What every salt backend needs and none of them should write twice.
//!
//! A salt kernel hands back one slot per hit: the salt it rebuilt and the
//! address it hashed. Turning that into a [`Hit`] re-derives the address from
//! the salt on the CPU, because the kernel reconstructs the salt separately
//! from the hashing path — so a drift between the two is otherwise invisible.
//! The address reported would be a real one, just not the one that salt
//! produces, and the user would deploy to somewhere else entirely.
//!
//! Both halves of that comparison arrive from the device, so it attests that
//! the pair is self-consistent rather than that either is what was asked for.
//! In 1nft, where the reportable result is a magic and the deployer rebuilds
//! the salt around it, that is not enough on its own: a salt whose low half is
//! not the pinned account digest passes it and still names an address the
//! magic cannot mint. `SaltConfig::salt_binds_to_mint_for` is the other half of
//! the check, and unlike the re-derivation it is not something `--no-verify`
//! turns off.
//!
//! The slot layout and the two ways of reading a round out of it live here for
//! the reason [`crate::profanity`] gives for the same split: a backend that
//! reads the layout subtly wrong reports another work item's salt, and two
//! copies of that accounting would fail differently. What stays per-backend is
//! how candidates are enumerated, and the kernels' own derivations stay written
//! twice on purpose, since that is what the agreement tests compare.

use miner_core::{Address, Salt, SaltConfig, ScoreSpec};

use crate::{EXACT_CAPACITY, Hit, Job, MAX_SCORE, Progress};

/// One result slot, matching `result` in kernels/opencl/salt.cl and `Result` in
/// kernels/metal/salt.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SaltSlot {
    pub salt: Salt,
    pub hash: Address,
    pub found: u32,
}

/// What turning a result slot into a `Hit` needs beyond the slot itself.
pub struct SaltRound<'a> {
    pub cfg: &'a SaltConfig,
    pub job: &'a Job,
    pub device_index: usize,
}

impl SaltRound<'_> {
    /// Build the hit a filled result slot describes.
    fn hit_from(&self, slot: &SaltSlot, score: u32, pattern: Option<usize>) -> Hit {
        let salt = slot.salt;

        Hit {
            score,
            address: slot.hash,
            salt: Some(salt),
            magic: self.cfg.magic_of(&salt),
            offset: None,
            pattern,
            device_index: self.device_index,
            verified: self.cfg.salt_binds_to_mint_for(&salt)
                && (!self.job.verify || self.cfg.address_for_salt(&salt) == slot.hash),
        }
    }

    /// Result slots are indexed by score, so the best hit is the highest
    /// occupied slot above what has already been reported.
    pub fn take_best(&self, results: &[SaltSlot], threshold: u64) -> Option<(u32, Hit)> {
        for score in (1..=MAX_SCORE).rev() {
            if results[score].found == 0 {
                continue;
            }
            if score as u64 <= threshold {
                break;
            }
            return Some((
                score as u32,
                self.hit_from(&results[score], score as u32, None),
            ));
        }
        None
    }

    /// Slots are the round's matches in arrival order; `total` is how many
    /// there were, including any the buffer had no room for. It is a separate
    /// argument because OpenCL counts in slot 0 of the result buffer and Metal
    /// in slot 0 of its atomic flag buffer.
    pub fn drain_exact(&self, results: &[SaltSlot], total: u32) -> Vec<Progress> {
        let stored = (total as usize).min(EXACT_CAPACITY);
        let masks = self.job.exact.as_deref().unwrap_or_default();

        let mut found: Vec<Progress> = (1..=stored)
            .map(|slot| {
                // `found` names the mask that matched in this layout, one-based
                // so an untouched slot is distinguishable from mask 0.
                let pattern = results[slot].found.saturating_sub(1) as usize;
                let score = masks.get(pattern).map_or(0, ScoreSpec::constrained_bytes);
                Progress::Hit(self.hit_from(&results[slot], score, Some(pattern)))
            })
            .collect();

        if let Some(dropped) = total.checked_sub(stored as u32).filter(|d| *d > 0) {
            found.push(Progress::Dropped {
                count: dropped,
                device_index: self.device_index,
            });
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use miner_core::{DEFAULT_PROXY_CODE_HASH, MineMode, ModeConfig, parse_address};

    use super::*;
    use crate::{KeccakVariant, RESULT_SLOTS, Tuning};

    fn config() -> SaltConfig {
        let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        SaltConfig::new(
            MineMode::Create2,
            deployer,
            DEFAULT_PROXY_CODE_HASH,
            [7u8; 32],
            None,
        )
        .unwrap()
    }

    fn job(cfg: &SaltConfig, exact: Option<Vec<ScoreSpec>>) -> Job {
        Job {
            mode: ModeConfig::Salt(cfg.clone()),
            score: ScoreSpec::zeros(),
            keccak: KeccakVariant::Tuned,
            tuning: Tuning::default(),
            duration: None,
            verify: true,
            exact,
        }
    }

    fn round<'a>(cfg: &'a SaltConfig, job: &'a Job) -> SaltRound<'a> {
        SaltRound {
            cfg,
            job,
            device_index: 0,
        }
    }

    /// A slot holding a salt and the address it really derives, so the
    /// re-derivation the drains do has something true to agree with.
    fn filled(gid: u32, cfg: &SaltConfig) -> SaltSlot {
        let salt = cfg.salt_at(0, gid, 1);
        SaltSlot {
            salt,
            hash: cfg.address_for_salt(&salt),
            found: 1,
        }
    }

    /// One occupied slot, indexed by score as the scoring kernel writes it.
    fn results(score: usize, cfg: &SaltConfig) -> Vec<SaltSlot> {
        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        slots[score] = filled(1, cfg);
        slots
    }

    fn hits(found: &[Progress]) -> Vec<&Hit> {
        found
            .iter()
            .filter_map(|p| match p {
                Progress::Hit(hit) => Some(hit),
                Progress::Dropped { .. } => None,
            })
            .collect()
    }

    /// Both kernels declare this layout, and a mismatch would be read as a
    /// salt spliced together from two neighbouring slots rather than as an
    /// error.
    #[test]
    fn the_wire_layout_is_what_the_kernels_declare() {
        assert_eq!(size_of::<SaltSlot>(), 56);
    }

    /// The round loop reloads its kernel's bar from a shared atomic, which only
    /// carries every device's progress because the caller publishes the score
    /// this hands back.
    #[test]
    fn the_best_slot_above_the_bar_is_the_one_reported() {
        let cfg = config();
        let job = job(&cfg, None);

        let (score, hit) = round(&cfg, &job)
            .take_best(&results(9, &cfg), 0)
            .expect("an occupied slot above the bar");
        assert_eq!((score, hit.score), (9, 9));
        assert!(hit.verified);
    }

    /// A device another has already beaten reports nothing, which is the round
    /// where its kernel used to carry on writing results the host reads and
    /// throws away.
    #[test]
    fn a_beaten_hit_is_not_reported_again() {
        let cfg = config();
        let job = job(&cfg, None);

        assert!(round(&cfg, &job).take_best(&results(9, &cfg), 12).is_none());
    }

    /// Every match in the round comes back, not the one the buffer happened to
    /// keep first. This is the whole point of the exact path: the scoring
    /// layout has one slot per score, and in `--exact` every match scores the
    /// same, so all but one used to be unrecoverable.
    #[test]
    fn the_exact_drain_returns_every_match_in_the_round() {
        let cfg = config();
        let job = job(&cfg, Some(vec![ScoreSpec::matching("00").unwrap()]));

        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        for (i, gid) in (1..=5u32).enumerate() {
            slots[i + 1] = filled(gid, &cfg);
        }

        let found = round(&cfg, &job).drain_exact(&slots, 5);
        let hits = hits(&found);
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.verified));
        // Each slot carries its own work item's salt rather than a repeat of
        // one, which is what a drain reading the wrong index would produce.
        let salts: std::collections::HashSet<_> = hits.iter().map(|hit| hit.salt).collect();
        assert_eq!(salts.len(), 5);
        assert!(!found.iter().any(|p| matches!(p, Progress::Dropped { .. })));
    }

    /// The count covers every match, including the ones there was no room for,
    /// so a round that overflows can say by how much instead of quietly losing
    /// it.
    #[test]
    fn the_exact_drain_reports_what_would_not_fit() {
        let cfg = config();
        let job = job(&cfg, Some(vec![ScoreSpec::matching("00").unwrap()]));

        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        for (gid, slot) in slots
            .iter_mut()
            .enumerate()
            .take(EXACT_CAPACITY + 1)
            .skip(1)
        {
            *slot = filled(gid as u32, &cfg);
        }

        let found = round(&cfg, &job).drain_exact(&slots, EXACT_CAPACITY as u32 + 44);
        assert_eq!(hits(&found).len(), EXACT_CAPACITY);
        assert!(matches!(
            found.last(),
            Some(Progress::Dropped { count: 44, .. })
        ));
    }

    /// Which mask matched is what identifies a hit here, since every match
    /// scores the same. The kernel stores it one-based so an untouched slot is
    /// not mistaken for a match on the first mask.
    #[test]
    fn the_exact_drain_names_the_mask_that_matched() {
        let cfg = config();
        let job = job(
            &cfg,
            Some(vec![
                ScoreSpec::matching("dead").unwrap(),
                ScoreSpec::matching("beefbeef").unwrap(),
            ]),
        );

        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        slots[1] = filled(1, &cfg);
        slots[1].found = 1;
        slots[2] = filled(2, &cfg);
        slots[2].found = 2;

        let found = round(&cfg, &job).drain_exact(&slots, 2);
        let hits = hits(&found);
        assert_eq!(hits[0].pattern, Some(0));
        assert_eq!(hits[1].pattern, Some(1));
        // The score is the mask's own constrained-byte count, so it stays
        // meaningful rather than repeating one number for every mask.
        assert_eq!((hits[0].score, hits[1].score), (2, 4));
    }

    /// 1nft reports the high half of the salt as the magic the deployer takes,
    /// and the other two salt modes have none.
    #[test]
    fn only_nft_carries_a_magic() {
        let account = parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();

        for (mode, mint_for) in [
            (MineMode::Create2, None),
            (MineMode::Create3, None),
            (MineMode::Nft, Some(account)),
        ] {
            let cfg = SaltConfig::new(mode, deployer, DEFAULT_PROXY_CODE_HASH, [7u8; 32], mint_for)
                .unwrap();
            let job = job(&cfg, None);
            let (_, hit) = round(&cfg, &job)
                .take_best(&results(4, &cfg), 0)
                .expect("an occupied slot");

            let salt = hit.salt.expect("a salt mode reports its salt");
            match mode {
                MineMode::Nft => assert_eq!(&hit.magic.expect("1nft reports a magic"), &salt[..16]),
                _ => assert!(hit.magic.is_none()),
            }
        }
    }

    /// A 1nft config whose deployer and account are the ones the report used,
    /// so the addresses below are the report's own.
    fn nft_config() -> SaltConfig {
        SaltConfig::new(
            MineMode::Nft,
            parse_address("0x1111111111111111111111111111111111111111").unwrap(),
            DEFAULT_PROXY_CODE_HASH,
            [7u8; 32],
            Some(parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap()),
        )
        .unwrap()
    }

    /// The salt a hostile device would send: an arbitrary low half, and the
    /// address that salt genuinely derives, so the two agree with each other.
    fn unbound_slot(cfg: &SaltConfig) -> SaltSlot {
        let mut salt = [b'A'; 32];
        salt[..16].copy_from_slice(&hex::decode("deadbeefdeadbeefdeadbeefdeadbeef").unwrap());
        SaltSlot {
            salt,
            hash: cfg.address_for_salt(&salt),
            found: 1,
        }
    }

    /// Reported by Kvazar: a device can pick the half of the salt that 1nft
    /// pins, and the address check alone accepts it, because both sides of that
    /// comparison come from the device. What is printed is then a magic that
    /// mints an address other than the one beside it.
    #[test]
    fn nft_rejects_a_salt_not_bound_to_the_account() {
        let cfg = nft_config();
        let job = job(&cfg, None);

        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        slots[9] = unbound_slot(&cfg);

        let (_, hit) = round(&cfg, &job)
            .take_best(&slots, 0)
            .expect("an occupied slot");

        assert!(!hit.verified);

        // Why it has to be rejected: the salt does derive the address reported,
        // so nothing else in the pipeline would notice, and the magic mints
        // somewhere else.
        let magic = hit.magic.expect("1nft reports a magic");
        let account = cfg.mint_for.unwrap();
        assert_eq!(cfg.address_for_salt(&hit.salt.unwrap()), hit.address);
        assert_ne!(
            cfg.address_for_salt(&miner_core::nft_salt(&magic, &account)),
            hit.address
        );
    }

    /// `--no-verify` trades the per-hit re-derivation for speed. The binding is
    /// not that trade: it is a single keccak of 20 bytes, and a magic that
    /// cannot mint is malformed rather than merely unchecked.
    #[test]
    fn the_account_binding_survives_no_verify() {
        let cfg = nft_config();
        let mut job = job(&cfg, None);
        job.verify = false;

        let mut slots = vec![SaltSlot::default(); RESULT_SLOTS];
        slots[9] = unbound_slot(&cfg);

        let (_, hit) = round(&cfg, &job)
            .take_best(&slots, 0)
            .expect("an occupied slot");
        assert!(!hit.verified);

        // And it still passes everything an honest device sends, which is the
        // half of this that a check rejecting too much would fail.
        let honest = cfg.salt_at(0, 1, 1);
        slots[9] = SaltSlot {
            salt: honest,
            hash: cfg.address_for_salt(&honest),
            found: 1,
        };
        let (_, hit) = round(&cfg, &job)
            .take_best(&slots, 0)
            .expect("an occupied slot");
        assert!(hit.verified);
    }
}
