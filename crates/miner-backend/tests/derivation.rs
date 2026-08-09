//! Cross-backend derivation tests.
//!
//! The unit tests in `miner-core` pin the address maths against known answers.
//! These tests pin the *backends* against that maths, which is the failure mode
//! that matters: a kernel that hashes the wrong pre-image still produces a
//! well-formed address, and nothing but a comparison will notice.
//!
//! GPU-dependent tests skip rather than fail when no device is present, so the
//! suite still runs in CI and in a container without a GPU.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use miner_backend::{Backend, Hit, Job, KeccakVariant, Reporter, Tuning, cpu::CpuBackend};
use miner_core::{
    DEFAULT_PROXY_CODE_HASH, MineMode, ModeConfig, ProfanityConfig, SaltConfig, ScoreSpec,
    keccak256, nft_salt, parse_address, secp256k1::generator,
};

#[derive(Default)]
struct Collector {
    hits: Mutex<Vec<Hit>>,
}

impl Reporter for &Collector {
    fn on_hit(&mut self, hit: &Hit) {
        self.hits.lock().unwrap().push(hit.clone());
    }
    fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
}

/// Run a backend until it reports a score of `stop_at` or the job's duration
/// expires, and return the hits in the order they arrived. `u32::MAX` means
/// "let the whole duration run".
fn run_until(backend: &mut dyn Backend, job: &Job, stop_at: u32) -> Vec<Hit> {
    struct Watch<'a> {
        hits: &'a Collector,
        found: &'a AtomicBool,
        stop_at: u32,
    }
    impl Reporter for Watch<'_> {
        fn on_hit(&mut self, hit: &Hit) {
            if hit.score >= self.stop_at {
                self.found.store(true, Ordering::SeqCst);
            }
            self.hits.hits.lock().unwrap().push(hit.clone());
        }
        fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
    }

    let collector = Collector::default();
    let found = AtomicBool::new(false);
    let stop = || found.load(Ordering::SeqCst);
    backend
        .run(
            job,
            &mut Watch {
                hits: &collector,
                found: &found,
                stop_at,
            },
            &stop,
        )
        .expect("backend run failed");

    let hits = collector.hits.lock().unwrap();
    hits.clone()
}

fn config(mode: MineMode) -> SaltConfig {
    let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
    let caller = (mode == MineMode::Nft)
        .then(|| parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap());
    let mut base = [0u8; 32];
    for (i, b) in base.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(29).wrapping_add(7);
    }
    SaltConfig::new(mode, deployer, DEFAULT_PROXY_CODE_HASH, base, caller).unwrap()
}

/// Plant a target the backend must find, then confirm the reported salt and
/// address are exactly the ones the CPU predicted for that work item.
fn planted_target(mut backend: Box<dyn Backend>, mode: MineMode, round_size: usize) -> Hit {
    const TARGET_GID: u32 = 4_242;
    const ROUND: u32 = 1;

    let cfg = config(mode);
    let target_salt = cfg.salt_at(0, TARGET_GID, ROUND);
    let target_address = cfg.address_for_salt(&target_salt);

    let job = Job {
        mode: ModeConfig::Salt(cfg),
        score: ScoreSpec::matching(&hex::encode(target_address)).unwrap(),
        keccak: KeccakVariant::Tuned,
        tuning: Tuning {
            round_size,
            work_size: 64,
            no_cache: true,
            ..Tuning::default()
        },
        duration: Some(Duration::from_secs(30)),
        verify: true,
        exact_score: None,
    };

    let hits = run_until(&mut *backend, &job, 20);
    let best = hits
        .iter()
        .max_by_key(|h| h.score)
        .unwrap_or_else(|| panic!("{} backend reported no hits at all", mode.as_str()))
        .clone();

    assert_eq!(
        best.score,
        20,
        "{}: planted target not reached",
        mode.as_str()
    );
    assert_eq!(
        best.address,
        target_address,
        "{}: wrong address",
        mode.as_str()
    );
    assert_eq!(
        best.salt,
        Some(target_salt),
        "{}: wrong salt",
        mode.as_str()
    );
    assert!(
        best.verified,
        "{}: hit failed CPU re-derivation",
        mode.as_str()
    );
    best
}

#[test]
fn cpu_finds_planted_target_in_every_salt_mode() {
    for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
        planted_target(Box::new(CpuBackend::new(Some(2))), mode, 1 << 13);
    }
}

#[test]
fn cpu_nft_hit_carries_a_usable_magic() {
    let hit = planted_target(Box::new(CpuBackend::new(Some(2))), MineMode::Nft, 1 << 13);
    let magic = hit.magic.expect("1nft hits must report a magic");
    let caller = parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap();

    // The deployer rebuilds the salt from the magic and the caller, so that
    // reconstruction has to match the salt actually mined.
    assert_eq!(nft_salt(&magic, &caller), hit.salt.unwrap());
    // And the low half must be the pinned caller digest, not mined bytes.
    assert_eq!(&hit.salt.unwrap()[16..], &keccak256(&caller)[16..32]);
}

/// `--exact` must keep reporting full matches instead of climbing to a best
/// score, and every reported address must satisfy the whole mask.
fn exact_matches(mut backend: Box<dyn Backend>, round_size: usize) {
    let cfg = config(MineMode::Create2);
    // One constrained nibble is frequent enough to hit many times quickly.
    let spec = ScoreSpec::matching("a").unwrap();
    let needed = spec.constrained_bytes();

    let job = Job {
        mode: ModeConfig::Salt(cfg),
        score: spec,
        keccak: KeccakVariant::Tuned,
        tuning: Tuning {
            round_size,
            work_size: 64,
            no_cache: true,
            ..Tuning::default()
        },
        duration: Some(Duration::from_secs(3)),
        verify: true,
        exact_score: Some(needed),
    };
    assert!(job.is_exact());
    assert_eq!(job.initial_threshold(), needed - 1);

    let collector = Collector::default();
    let stop = || false;
    backend
        .run(&job, &mut &collector, &stop)
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
    assert!(
        hits.len() > 1,
        "exact mode should report every match, got {}",
        hits.len()
    );
    for hit in hits.iter() {
        // Never a partial match, and never a climbing score.
        assert_eq!(hit.score, needed, "exact mode reported a partial match");
        assert!(hit.verified, "exact mode hit failed CPU re-derivation");
        // The mask constrains the high nibble of byte 0 to 0xa.
        assert_eq!(hit.address[0] >> 4, 0xa);
    }
}

#[test]
fn exact_mode_reports_repeated_full_matches() {
    exact_matches(Box::new(CpuBackend::new(Some(2))), 1 << 12);
}

/// The generator as the seed public key: its private half is 1, so a failing
/// case here can be reproduced by hand.
fn profanity_config(contract: bool) -> ProfanityConfig {
    ProfanityConfig {
        seed_public_key: generator(),
        contract,
    }
}

fn profanity_job(cfg: ProfanityConfig, score: ScoreSpec, seconds: u64) -> Job {
    Job {
        mode: ModeConfig::Profanity(cfg),
        score,
        keccak: KeccakVariant::Tuned,
        tuning: Tuning::default(),
        duration: Some(Duration::from_secs(seconds)),
        verify: true,
        exact_score: None,
    }
}

/// The CPU profanity loop reports each hit as it finds it, so every hit has to
/// arrive exactly once. Re-reporting the newest one on each speed poll — which
/// is what passing `len - 1` to a drain does — shows up here as a repeated
/// offset and a score that stops climbing.
#[test]
fn cpu_profanity_reports_each_hit_once() {
    let job = profanity_job(profanity_config(false), ScoreSpec::zeros(), 2);
    let hits = run_until(&mut CpuBackend::new(Some(1)), &job, u32::MAX);
    assert!(!hits.is_empty(), "no hits, so nothing here was checked");

    let mut previous = 0;
    let mut offsets = HashSet::new();
    for hit in hits.iter() {
        assert!(
            hit.score > previous,
            "profanity scores must strictly improve"
        );
        previous = hit.score;
        let offset = hit.offset.expect("a profanity hit carries an offset");
        assert!(offsets.insert(offset), "an offset was reported twice");
    }
}

/// Two runs against one public key must not walk the same offsets, or a second
/// attempt adds nothing to a search and two people holding the same published
/// seed key find the same addresses. From a fixed start both runs report the
/// same first offset, which is what this catches.
#[test]
fn cpu_profanity_starts_somewhere_new_each_run() {
    let job = profanity_job(profanity_config(false), ScoreSpec::zeros(), 5);
    let first = run_until(&mut CpuBackend::new(Some(1)), &job, 1);
    let second = run_until(&mut CpuBackend::new(Some(1)), &job, 1);

    let first = first.first().expect("the first run reported no hit");
    let second = second.first().expect("the second run reported no hit");
    assert_ne!(
        first.offset, second.offset,
        "both runs started from the same offset"
    );
    assert_ne!(first.address, second.address);
}

/// The walk starts at exactly the offset it is handed and reports that offset
/// for its first candidate. This is the contract the cross-backend agreement
/// test depends on, so it is pinned separately from the test that uses it.
#[test]
fn cpu_profanity_starts_at_the_offset_it_is_given() {
    let cfg = profanity_config(false);
    // Top two bytes clear, as a drawn offset has them.
    let mut base = [0u8; 32];
    for (i, b) in base.iter_mut().enumerate().skip(2) {
        *b = (i as u8).wrapping_mul(43).wrapping_add(3);
    }
    let expected = cfg
        .address_for_offset(&base)
        .expect("the offset must name a point");

    let score = ScoreSpec::matching(&hex::encode(expected)).unwrap();
    let job = profanity_job(cfg, score, 10);
    let hits = run_until(&mut CpuBackend::starting_at(Some(1), base), &job, 20);

    let hit = hits
        .first()
        .expect("the very first candidate is the one asked for");
    assert_eq!(hit.score, 20, "the walk did not begin at the given offset");
    assert_eq!(hit.address, expected);
    assert_eq!(hit.offset, Some(base));
    assert!(hit.verified);
}

/// A profanity hit names its address with an offset the walk accounts for
/// separately from the point it actually reached — the same split that makes
/// the salt modes' `salt_at` worth testing. Nothing inside the walk would
/// notice the two drifting apart, so every reported offset is re-derived here
/// the long way round, through a double-and-add ladder rather than a walk.
#[test]
fn cpu_profanity_offsets_re_derive_to_the_address_reported() {
    for contract in [false, true] {
        let cfg = profanity_config(contract);
        let job = profanity_job(cfg.clone(), ScoreSpec::zeros(), 2);
        let hits = run_until(&mut CpuBackend::new(Some(1)), &job, u32::MAX);
        assert!(!hits.is_empty(), "no hits, so nothing here was checked");

        for hit in &hits {
            let offset = hit.offset.expect("a profanity hit carries an offset");
            assert_eq!(
                cfg.address_for_offset(&offset),
                Some(hit.address),
                "contract={contract}: the offset names a different address"
            );
            assert!(hit.verified, "contract={contract}: hit reported unverified");
        }
    }
}

/// Without `--exact` the bar climbs, so each reported score is strictly better
/// than the one before it.
#[test]
fn ordinary_scoring_reports_only_improvements() {
    let cfg = config(MineMode::Create2);
    let job = Job {
        mode: ModeConfig::Salt(cfg),
        score: ScoreSpec::zeros(),
        keccak: KeccakVariant::Tuned,
        tuning: Tuning {
            round_size: 1 << 12,
            work_size: 64,
            no_cache: true,
            ..Tuning::default()
        },
        duration: Some(Duration::from_secs(2)),
        verify: true,
        exact_score: None,
    };

    let collector = Collector::default();
    let stop = || false;
    let mut backend = CpuBackend::new(Some(2));
    backend
        .run(&job, &mut &collector, &stop)
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
    let mut previous = 0;
    for hit in hits.iter() {
        assert!(hit.score > previous, "scores must strictly improve");
        previous = hit.score;
    }
}

#[cfg(feature = "opencl")]
mod opencl {
    use super::*;
    use miner_backend::opencl::{profanity::ProfanityBackend, salt::SaltBackend};

    fn backend() -> Option<Box<dyn Backend>> {
        match SaltBackend::new(&[]) {
            Ok(b) => Some(Box::new(b)),
            Err(e) => {
                eprintln!("skipping OpenCL test: {e}");
                None
            }
        }
    }

    fn profanity_backend() -> Option<ProfanityBackend> {
        match ProfanityBackend::new(&[]) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping OpenCL profanity test: {e}");
                None
            }
        }
    }

    /// A profanity job small enough to be a test. 255 x 64 is 16320 points
    /// against the default 4.2M, so the three scratch buffers and the init
    /// phase take a moment rather than 400 MB and several seconds.
    fn small_profanity_job(cfg: ProfanityConfig, seconds: u64) -> Job {
        Job {
            tuning: Tuning {
                inverse_size: 255,
                inverse_multiple: 64,
                work_size: 64,
                no_cache: true,
                ..Tuning::default()
            },
            ..profanity_job(cfg, ScoreSpec::zeros(), seconds)
        }
    }

    /// The secp256k1 kernel — modular inversion, batched-inverse point
    /// addition, and a documented set of deliberately unhandled edge cases — is
    /// the most intricate code here and was the only kernel with no agreement
    /// test. Until now its correctness rested on `--verify`, which fires on a
    /// real hit and so tells an operator only after the fact.
    ///
    /// No target can be planted: unlike a salt work item, whose address
    /// `salt_at` predicts before the run, a profanity work item's address is
    /// not addressable in advance. So the assertion is the one that protects a
    /// real run — every offset the kernel reports has to name the address the
    /// kernel reported with it — made whenever the suite is run rather than
    /// hours into someone's rental.
    #[test]
    fn opencl_profanity_offsets_name_the_addresses_reported() {
        // --contract runs a second keccak over the account address, which is a
        // separate path through the kernel and had no coverage either.
        for contract in [false, true] {
            let Some(mut b) = profanity_backend() else {
                return;
            };
            let cfg = profanity_config(contract);
            let job = small_profanity_job(cfg.clone(), 3);
            let hits = run_until(&mut b, &job, u32::MAX);

            assert!(
                !hits.is_empty(),
                "contract={contract}: no hits, so nothing here was checked"
            );
            for hit in &hits {
                let offset = hit.offset.expect("a profanity hit carries an offset");
                assert_eq!(
                    cfg.address_for_offset(&offset),
                    Some(hit.address),
                    "contract={contract}: the offset names a different address"
                );
                assert!(hit.verified, "contract={contract}: hit reported unverified");
            }
        }
    }

    /// The strongest claim this path can make, and the one the salt modes
    /// already enjoy: two implementations agreeing, rather than one checked
    /// against itself.
    ///
    /// The kernel's field arithmetic is written in OpenCL C and shares no code
    /// with the host's. Handing an offset it reported to the CPU walk as a
    /// starting point makes the two derive an address for the same scalar by
    /// entirely separate routes.
    #[test]
    fn the_kernel_and_the_cpu_walk_agree_on_one_offset() {
        let Some(mut gpu) = profanity_backend() else {
            return;
        };
        let cfg = profanity_config(false);
        let hits = run_until(&mut gpu, &small_profanity_job(cfg.clone(), 10), 1);
        let hit = hits.first().expect("the kernel reported no hits at all");
        let offset = hit.offset.expect("a profanity hit carries an offset");

        // Mask on the whole address the kernel claims for that offset. The CPU
        // examines the offset it is started at first, so either its very first
        // candidate is the answer or nothing later will be.
        let score = ScoreSpec::matching(&hex::encode(hit.address)).unwrap();
        let job = profanity_job(cfg, score, 10);
        let walked = run_until(&mut CpuBackend::starting_at(Some(1), offset), &job, 20);

        let found = walked
            .first()
            .expect("the CPU walk reported nothing at all for the kernel's offset");
        assert_eq!(
            found.offset,
            Some(offset),
            "the CPU walk did not start where it was told"
        );
        assert_eq!(
            found.address, hit.address,
            "the kernel and the CPU walk derived different addresses for one offset"
        );
        assert_eq!(found.score, 20);
    }

    #[test]
    fn opencl_agrees_with_the_cpu_in_every_salt_mode() {
        for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
            let Some(b) = backend() else { return };
            planted_target(b, mode, 1 << 16);
        }
    }

    /// Every match in an `--exact` round targets the same result slot, and a
    /// loose mask puts thousands of work items through it. That is the workload
    /// where a truncated first-writer check lets two of them interleave their
    /// writes, so the salt in the slot belongs to one and the address to
    /// another and re-derivation disagrees.
    #[test]
    fn opencl_exact_mode_reports_untorn_full_matches() {
        let Some(b) = backend() else { return };
        exact_matches(b, 1 << 16);
    }

    /// A round size that `--work` does not divide leaves a chunk no local size
    /// fits. The launch has to fall back to a driver-chosen size instead of
    /// failing the whole run with CL_INVALID_WORK_GROUP_SIZE.
    #[test]
    fn opencl_runs_a_round_the_work_size_does_not_divide() {
        let Some(mut b) = backend() else { return };

        let job = Job {
            mode: ModeConfig::Salt(config(MineMode::Create2)),
            score: ScoreSpec::zeros(),
            keccak: KeccakVariant::Tuned,
            tuning: Tuning {
                round_size: (1 << 16) + 1,
                work_size: 64,
                no_cache: true,
                ..Tuning::default()
            },
            duration: Some(Duration::from_secs(2)),
            verify: true,
            exact_score: None,
        };

        let collector = Collector::default();
        let stop = || false;
        b.run(&job, &mut &collector, &stop)
            .expect("a round the work size does not divide must still run");
    }

    /// The two Keccak variants are meant to be interchangeable, so they must
    /// find the same address for the same work item.
    #[test]
    fn both_keccak_variants_produce_the_same_address() {
        let Some(_) = backend() else { return };

        let cfg = config(MineMode::Create3);
        let target_salt = cfg.salt_at(0, 999, 1);
        let target = cfg.address_for_salt(&target_salt);

        for keccak in KeccakVariant::all() {
            let Ok(mut b) = SaltBackend::new(&[]) else {
                return;
            };
            let job = Job {
                mode: ModeConfig::Salt(cfg.clone()),
                score: ScoreSpec::matching(&hex::encode(target)).unwrap(),
                keccak: *keccak,
                tuning: Tuning {
                    round_size: 1 << 16,
                    work_size: 64,
                    no_cache: true,
                    ..Tuning::default()
                },
                duration: Some(Duration::from_secs(30)),
                verify: true,
                exact_score: None,
            };

            let hits = run_until(&mut b, &job, 20);
            let best = hits.iter().max_by_key(|h| h.score).expect("no hits");
            assert_eq!(
                best.address,
                target,
                "keccak variant {} disagreed",
                keccak.as_str()
            );
            assert_eq!(best.salt, Some(target_salt));
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use miner_backend::metal::MetalBackend;

    #[test]
    fn metal_agrees_with_the_cpu_in_every_salt_mode() {
        for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
            match MetalBackend::new() {
                Ok(b) => {
                    planted_target(Box::new(b), mode, 1 << 16);
                }
                Err(e) => {
                    eprintln!("skipping Metal test: {e}");
                    return;
                }
            }
        }
    }
}
