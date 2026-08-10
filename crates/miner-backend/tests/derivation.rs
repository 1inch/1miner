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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use miner_backend::{
    Backend, EXACT_CAPACITY, Hit, Job, KeccakVariant, Reporter, Tuning, cpu::CpuBackend,
};
use miner_core::{
    DEFAULT_PROXY_CODE_HASH, MineMode, ModeConfig, ProfanityConfig, SaltConfig, ScoreSpec,
    keccak256, nft_salt, parse_address, secp256k1::generator,
};

#[derive(Default)]
struct Collector {
    hits: Mutex<Vec<Hit>>,
    /// Candidates tried over the whole run, from the closing summary. Divided
    /// by the round size this is the number of rounds, which is what bounds
    /// how many hits a one-slot-per-score buffer could have reported.
    hashes: AtomicU64,
    dropped: AtomicU64,
}

impl Reporter for &Collector {
    fn on_hit(&mut self, hit: &Hit) {
        self.hits.lock().unwrap().push(hit.clone());
    }
    fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
    fn on_summary(&mut self, summary: &miner_backend::speed::SpeedSummary) {
        self.hashes.store(summary.hashes, Ordering::Relaxed);
    }
    fn on_dropped(&mut self, count: u32, _device_index: usize) {
        self.dropped.fetch_add(u64::from(count), Ordering::Relaxed);
    }
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
        exact: None,
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

/// `--exact` must report every full match, and every reported address must
/// satisfy the whole mask.
///
/// The count is the assertion that matters. One constrained nibble matches
/// about one candidate in sixteen, so a round of `round_size` produces roughly
/// `round_size / 16` matches and the run produces that many times the number of
/// rounds. The one-slot-per-score layout could only ever return one per round,
/// so requiring far more than the rounds could have produced is what separates
/// the append path from it — and a test asserting only that matches keep
/// arriving passed against either.
fn exact_matches(mut backend: Box<dyn Backend>, round_size: usize, capped: bool) {
    let cfg = config(MineMode::Create2);
    let spec = ScoreSpec::matching("a").unwrap();
    let needed = spec.constrained_bytes();

    let seconds = 3;
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
        duration: Some(Duration::from_secs(seconds)),
        verify: true,
        exact: Some(vec![spec]),
    };
    assert!(job.is_exact());

    let collector = Collector::default();
    let stop = || false;
    backend
        .run(&job, &mut &collector, &stop)
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
    let hashes = collector.hashes.load(Ordering::Relaxed);
    assert!(
        hashes > 0,
        "no closing summary, so there is nothing to compare the hit count against"
    );
    let rounds = (hashes / round_size as u64).max(1);
    // One constrained nibble matches about one candidate in sixteen, and a GPU
    // round hands back at most a bufferful of them.
    let expected = round_size as u64 / 16;
    let per_round = if capped {
        expected.min(EXACT_CAPACITY as u64)
    } else {
        expected
    };
    // Half the expected yield leaves room for the last round being cut short
    // and for the summary excluding a warmup, while still sitting far above the
    // one per round the scoring layout could manage.
    let floor = (rounds * per_round / 2).max(2);
    assert!(
        hits.len() as u64 >= floor,
        "exact mode should report every match in a round rather than one: got {} over {rounds} \
         rounds, where one per round would be {rounds} and this round size implies about \
         {per_round}",
        hits.len(),
    );

    // A round that finds more than it can keep has to say so, rather than
    // looking like one where matches were simply rarer than they were.
    if capped && expected > EXACT_CAPACITY as u64 {
        assert!(
            collector.dropped.load(Ordering::Relaxed) > 0,
            "a round of {round_size} yields about {expected} matches against a capacity of \
             {EXACT_CAPACITY}, so the overflow should have been reported"
        );
    }

    for hit in hits.iter() {
        // Never a partial match, and never a climbing score.
        assert_eq!(hit.score, needed, "exact mode reported a partial match");
        assert!(hit.verified, "exact mode hit failed CPU re-derivation");
        assert_eq!(hit.pattern, Some(0), "one mask means every hit matched it");
        // The mask constrains the high nibble of byte 0 to 0xa.
        assert_eq!(hit.address[0] >> 4, 0xa);
    }
}

#[test]
fn exact_mode_reports_repeated_full_matches() {
    exact_matches(Box::new(CpuBackend::new(Some(2))), 1 << 12, false);
}

/// Several masks in one pass, which the scoring layout cannot express at all:
/// it carries one mask and indexes results by score. Each hit has to name the
/// mask it matched, and each has to actually satisfy that mask.
fn exact_matches_several_masks(mut backend: Box<dyn Backend>, round_size: usize) {
    let cfg = config(MineMode::Create2);
    // Disjoint, so which mask matched is decided by the address rather than by
    // the order the kernel happens to test them in.
    let masks = vec![
        ScoreSpec::matching("a").unwrap(),
        ScoreSpec::matching("b").unwrap(),
        ScoreSpec::matching("c").unwrap(),
    ];

    let job = Job {
        mode: ModeConfig::Salt(cfg),
        score: masks[0],
        keccak: KeccakVariant::Tuned,
        tuning: Tuning {
            round_size,
            work_size: 64,
            no_cache: true,
            ..Tuning::default()
        },
        duration: Some(Duration::from_secs(3)),
        verify: true,
        exact: Some(masks.clone()),
    };

    let collector = Collector::default();
    let stop = || false;
    backend
        .run(&job, &mut &collector, &stop)
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
    assert!(!hits.is_empty(), "no hits in three seconds");

    let mut seen = [false; 3];
    for hit in hits.iter() {
        let pattern = hit.pattern.expect("an exact hit names the mask it matched");
        assert!(hit.verified);
        assert_eq!(
            hit.address[0] >> 4,
            0xa + pattern as u8,
            "hit reported against a mask it does not match"
        );
        seen[pattern] = true;
    }
    assert_eq!(seen, [true; 3], "every mask should have matched something");
}

#[test]
fn exact_mode_searches_several_masks_at_once() {
    exact_matches_several_masks(Box::new(CpuBackend::new(Some(2))), 1 << 12);
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
        exact: None,
    }
}

/// A profanity job small enough to be a test. 255 x 64 is 16320 points against
/// the default 4.2M, so the three scratch buffers and the init phase take a
/// moment rather than 400 MB and several seconds.
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

/// Every offset a backend reports has to name the address reported with it.
///
/// No target can be planted: unlike a salt work item, whose address `salt_at`
/// predicts before the run, a profanity work item's address is not addressable
/// in advance. So the assertion is the one that protects a real run, made
/// whenever the suite is run rather than hours into someone's rental.
///
/// This is also the check that catches round accounting drifting from the
/// walk, which is silent in every other way: the address is a real one and the
/// offset is well formed, it simply names the key to a different address.
fn profanity_offsets_name_their_addresses(
    backend: &mut dyn Backend,
    cfg: &ProfanityConfig,
    contract: bool,
) {
    let hits = run_until(backend, &small_profanity_job(cfg.clone(), 3), u32::MAX);

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

/// `--exact` in profanity, where the mask is checked against an address the
/// kernel derives through secp256k1 rather than keccak alone, and the result is
/// an offset rather than a salt.
///
/// The OpenCL kernel behind this had never been executed before: it was in the
/// source, compiled on every run, and never selected. Its comparison is the
/// part to distrust, so what is asserted is that every address reported really
/// does satisfy the mask it was reported against, and re-derives from its own
/// offset.
///
/// `digits` sets how rare a match is, because every hit costs a full scalar
/// multiplication to verify. One nibble means one candidate in sixteen, which a
/// GPU produces faster than the host can re-derive them.
fn profanity_exact_matches(backend: &mut dyn Backend, seconds: u64, digits: usize) {
    let cfg = profanity_config(false);
    let masks = vec![
        ScoreSpec::matching(&"a".repeat(digits)).unwrap(),
        ScoreSpec::matching(&"b".repeat(digits)).unwrap(),
    ];
    let job = Job {
        mode: ModeConfig::Profanity(cfg.clone()),
        score: masks[0],
        keccak: KeccakVariant::Tuned,
        tuning: Tuning {
            inverse_size: 255,
            inverse_multiple: 64,
            work_size: 64,
            no_cache: true,
            ..Tuning::default()
        },
        duration: Some(Duration::from_secs(seconds)),
        verify: true,
        exact: Some(masks),
    };

    // Both masks matching is the assertion that catches a kernel testing only
    // the first, so the run ends as soon as both have rather than burning the
    // timeout. Each hit costs a scalar multiplication to verify, which is what
    // makes a fixed duration an unreliable way to collect enough of them.
    struct UntilBothMasks<'a> {
        inner: &'a Collector,
        seen: &'a AtomicU64,
    }
    impl Reporter for UntilBothMasks<'_> {
        fn on_hit(&mut self, hit: &Hit) {
            if let Some(pattern) = hit.pattern {
                self.seen.fetch_or(1 << pattern, Ordering::SeqCst);
            }
            (&mut &*self.inner).on_hit(hit);
        }
        fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
    }

    let collector = Collector::default();
    let seen = AtomicU64::new(0);
    let stop = || seen.load(Ordering::SeqCst) == 0b11;
    backend
        .run(
            &job,
            &mut UntilBothMasks {
                inner: &collector,
                seen: &seen,
            },
            &stop,
        )
        .expect("run failed");

    let hits = collector.hits.lock().unwrap();
    assert!(!hits.is_empty(), "no hits, so nothing here was checked");

    for hit in hits.iter() {
        let pattern = hit.pattern.expect("an exact hit names the mask it matched");
        assert_eq!(
            hit.address[0] >> 4,
            0xa + pattern as u8,
            "hit reported against a mask it does not match"
        );
        let offset = hit.offset.expect("a profanity hit carries an offset");
        assert_eq!(
            cfg.address_for_offset(&offset),
            Some(hit.address),
            "the offset names a different address"
        );
        assert!(hit.verified);
    }
    assert_eq!(
        seen.load(Ordering::SeqCst),
        0b11,
        "only one of the two masks ever matched in {seconds}s, so nothing here shows that the \
         second one is tested at all"
    );
}

#[test]
fn cpu_profanity_exact_matches_every_mask() {
    profanity_exact_matches(&mut CpuBackend::new(Some(2)), 30, 1);
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
        exact: None,
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

    /// The secp256k1 kernel — modular inversion, batched-inverse point
    /// addition, and a documented set of deliberately unhandled edge cases — is
    /// the most intricate code here and was the only kernel with no agreement
    /// test. Until now its correctness rested on `--verify`, which fires on a
    /// real hit and so tells an operator only after the fact.
    #[test]
    fn opencl_profanity_offsets_name_the_addresses_reported() {
        // --contract runs a second keccak over the account address, which is a
        // separate path through the kernel and had no coverage either.
        for contract in [false, true] {
            let Some(mut b) = profanity_backend() else {
                return;
            };
            let cfg = profanity_config(contract);
            profanity_offsets_name_their_addresses(&mut b, &cfg, contract);
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
        exact_matches(b, 1 << 16, true);
    }

    #[test]
    fn opencl_exact_mode_searches_several_masks_at_once() {
        let Some(b) = backend() else { return };
        exact_matches_several_masks(b, 1 << 16);
    }

    /// The exact kernel in profanity.cl, which until now was compiled on every
    /// run and never selected, so nothing had ever executed it.
    #[test]
    fn opencl_profanity_exact_matches_every_mask() {
        let Some(mut b) = profanity_backend() else {
            return;
        };
        profanity_exact_matches(&mut b, 30, 3);
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
            exact: None,
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
                exact: None,
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

    fn backend() -> Option<Box<dyn Backend>> {
        match MetalBackend::new() {
            Ok(b) => Some(Box::new(b)),
            Err(e) => {
                eprintln!("skipping Metal test: {e}");
                None
            }
        }
    }

    #[test]
    fn metal_agrees_with_the_cpu_in_every_salt_mode() {
        for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
            let Some(b) = backend() else { return };
            planted_target(b, mode, 1 << 16);
        }
    }

    #[test]
    fn metal_exact_mode_reports_every_match_in_the_round() {
        let Some(b) = backend() else { return };
        exact_matches(b, 1 << 16, true);
    }

    #[test]
    fn metal_exact_mode_searches_several_masks_at_once() {
        let Some(b) = backend() else { return };
        exact_matches_several_masks(b, 1 << 16);
    }

    fn profanity_backend() -> Option<MetalBackend> {
        match MetalBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping Metal profanity test: {e}");
                None
            }
        }
    }

    /// The Metal secp256k1 kernel against the CPU, exactly as the OpenCL one is
    /// checked. This is what caught the round accounting: the Metal loop reads
    /// each pass's results immediately, where the OpenCL loop reads them at the
    /// top of the next iteration, and the one-pass difference between the two
    /// is the difference between a usable key and someone else's address.
    #[test]
    fn metal_profanity_offsets_name_the_addresses_reported() {
        for contract in [false, true] {
            let Some(mut b) = profanity_backend() else {
                return;
            };
            let cfg = profanity_config(contract);
            profanity_offsets_name_their_addresses(&mut b, &cfg, contract);
        }
    }

    /// Two implementations agreeing, rather than one checked against itself.
    /// The kernel's field arithmetic is written in Metal Shading Language and
    /// shares no code with the host's, so handing an offset it reported to the
    /// CPU walk makes the two derive an address for one scalar by entirely
    /// separate routes.
    #[test]
    fn the_metal_kernel_and_the_cpu_walk_agree_on_one_offset() {
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

    /// The exact kernel, which shares the point walk with the scoring one but
    /// nothing else: a different result layout, a counter instead of a bar, and
    /// several masks tested per candidate.
    #[test]
    fn metal_profanity_exact_matches_every_mask() {
        let Some(mut b) = profanity_backend() else {
            return;
        };
        profanity_exact_matches(&mut b, 30, 3);
    }
}
