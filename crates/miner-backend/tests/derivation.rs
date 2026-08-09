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

    let collector = Collector::default();
    let found = AtomicBool::new(false);
    let stop = || found.load(Ordering::SeqCst);

    struct Watch<'a> {
        inner: &'a Collector,
        found: &'a AtomicBool,
    }
    impl Reporter for Watch<'_> {
        fn on_hit(&mut self, hit: &Hit) {
            if hit.score == 20 {
                self.found.store(true, Ordering::SeqCst);
            }
            self.inner.hits.lock().unwrap().push(hit.clone());
        }
        fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
    }

    backend
        .run(
            &job,
            &mut Watch {
                inner: &collector,
                found: &found,
            },
            &stop,
        )
        .expect("backend run failed");

    let hits = collector.hits.lock().unwrap();
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

/// The CPU profanity loop reports each hit as it finds it, so every hit has to
/// arrive exactly once. Re-reporting the newest one on each speed poll — which
/// is what passing `len - 1` to a drain does — shows up here as a repeated
/// offset and a score that stops climbing.
#[test]
fn cpu_profanity_reports_each_hit_once() {
    let job = Job {
        mode: ModeConfig::Profanity(ProfanityConfig {
            seed_public_key: generator(),
            contract: false,
        }),
        score: ScoreSpec::zeros(),
        keccak: KeccakVariant::Tuned,
        tuning: Tuning::default(),
        duration: Some(Duration::from_secs(2)),
        verify: true,
        exact_score: None,
    };

    let collector = Collector::default();
    let stop = || false;
    let mut backend = CpuBackend::new(Some(1));
    backend
        .run(&job, &mut &collector, &stop)
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
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
    use miner_backend::opencl::salt::SaltBackend;

    fn backend() -> Option<Box<dyn Backend>> {
        match SaltBackend::new(&[]) {
            Ok(b) => Some(Box::new(b)),
            Err(e) => {
                eprintln!("skipping OpenCL test: {e}");
                None
            }
        }
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

            let collector = Collector::default();
            let found = AtomicBool::new(false);
            struct Watch<'a> {
                inner: &'a Collector,
                found: &'a AtomicBool,
            }
            impl Reporter for Watch<'_> {
                fn on_hit(&mut self, hit: &Hit) {
                    if hit.score == 20 {
                        self.found.store(true, Ordering::SeqCst);
                    }
                    self.inner.hits.lock().unwrap().push(hit.clone());
                }
                fn on_speed(&mut self, _t: f64, _p: &[f64]) {}
            }
            let stop = || found.load(Ordering::SeqCst);
            b.run(
                &job,
                &mut Watch {
                    inner: &collector,
                    found: &found,
                },
                &stop,
            )
            .expect("run failed");

            let hits = collector.hits.lock().unwrap();
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
