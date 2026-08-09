//! `1miner self-test`: prove the selected backend derives the same addresses as
//! the CPU reference before a long run is trusted to it.
//!
//! The interesting check is the planted target. A work item is chosen, its
//! address computed on the CPU, and a full 20-byte mask built from that
//! address. The backend then has to find exactly that work item and report the
//! salt for it. Passing exercises the pre-image construction, the keccak
//! padding, the second CREATE hash, the scoring, and the separate salt
//! reconstruction path inside the kernel, all at once.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use miner_backend::{Hit, Job, Reporter, Tuning};
use miner_core::{
    DEFAULT_PROXY_CODE_HASH, MineMode, ModeConfig, PROXY_CHILD_BYTECODE, SaltConfig, ScoreSpec,
    create3_address, keccak256, parse_address,
};

use crate::cli::CommonArgs;

/// Small enough to finish instantly, large enough to span many work groups.
const ROUND: usize = 1 << 16;
const TARGET_GID: u32 = 12_345;
const ROUND_INDEX: u32 = 1;

#[derive(Default)]
struct Collector {
    hits: Vec<Hit>,
}

impl Reporter for Collector {
    fn on_hit(&mut self, hit: &Hit) {
        self.hits.push(hit.clone());
    }
    fn on_speed(&mut self, _total: f64, _per_device: &[f64]) {}
}

pub fn run(common: &CommonArgs) -> anyhow::Result<()> {
    let mut failures = 0usize;

    println!("Constants");
    check(
        &mut failures,
        "proxy bytecode hash",
        keccak256(&PROXY_CHILD_BYTECODE) == DEFAULT_PROXY_CODE_HASH,
    );

    println!("\nKnown-answer vectors (CPU)");
    let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf")?;
    let expected = parse_address("0x6c8ed9dc3734d7944beddd2fb5acdf5f17247870")?;
    check(
        &mut failures,
        "create3 zero-salt vector",
        create3_address(&factory, &[0u8; 32], &DEFAULT_PROXY_CODE_HASH) == expected,
    );

    println!("\nBackend agreement ({})", common.backend);
    for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
        match planted_target(mode, common) {
            Ok(true) => check(&mut failures, &format!("{} planted target", mode.as_str()), true),
            Ok(false) => check(&mut failures, &format!("{} planted target", mode.as_str()), false),
            Err(e) => {
                check(&mut failures, &format!("{} planted target", mode.as_str()), false);
                println!("      {e}");
            }
        }
    }

    println!();
    if failures > 0 {
        anyhow::bail!("{failures} self-test check(s) failed; do not trust this device");
    }
    println!("All checks passed.");
    Ok(())
}

fn check(failures: &mut usize, name: &str, ok: bool) {
    if ok {
        println!("  ok    {name}");
    } else {
        *failures += 1;
        println!("  FAIL  {name}");
    }
}

/// Ask the backend to find one work item whose address we already know.
fn planted_target(mode: MineMode, common: &CommonArgs) -> anyhow::Result<bool> {
    let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf")?;
    let caller = (mode == MineMode::Nft)
        .then(|| parse_address("0x00000000219ab540356cbb839cbe05303d7705fa"))
        .transpose()?;

    // A fixed base salt keeps the test reproducible.
    let mut base = [0u8; 32];
    for (i, b) in base.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let cfg = SaltConfig::new(mode, deployer, DEFAULT_PROXY_CODE_HASH, base, caller)?;

    let target_salt = cfg.salt_at(0, TARGET_GID, ROUND_INDEX);
    let target_address = cfg.address_for_salt(&target_salt);

    // Full 20-byte mask: only the planted work item can score the maximum.
    let score = ScoreSpec::matching(&hex::encode(target_address))?;

    let tuning = Tuning {
        work_size: common.work.min(64),
        work_max: None,
        round_size: ROUND,
        skip_devices: common.skip.clone(),
        // Constants differ per self-test run, so a cached binary would never
        // match anyway; skipping the cache keeps the run self-contained.
        no_cache: true,
        ..Tuning::default()
    };

    let job = Job {
        mode: ModeConfig::Salt(cfg.clone()),
        score,
        keccak: common.keccak(),
        tuning: tuning.clone(),
        // The planted item is reached in round 1; allow a little slack.
        duration: Some(std::time::Duration::from_secs(20)),
        verify: true,
        exact_score: None,
    };

    let mut backend = crate::open_backend(common, &job.mode, &tuning)?;
    let mut collector = Collector::default();
    let stop = Arc::new(AtomicBool::new(false));

    // Stop as soon as a perfect score arrives rather than burning the timeout.
    let found = Arc::new(AtomicBool::new(false));
    let should_stop = {
        let stop = Arc::clone(&stop);
        let found = Arc::clone(&found);
        move || stop.load(Ordering::SeqCst) || found.load(Ordering::SeqCst)
    };

    backend.run(&job, &mut WatchFor { inner: &mut collector, found: Arc::clone(&found) }, &should_stop)?;

    let best = collector.hits.iter().max_by_key(|h| h.score);
    Ok(match best {
        Some(hit) => {
            hit.address == target_address
                && hit.salt == Some(target_salt)
                && hit.verified
                && hit.score == 20
        }
        None => false,
    })
}

/// Flip a flag once the perfect score turns up so the run can end early.
struct WatchFor<'a> {
    inner: &'a mut Collector,
    found: Arc<AtomicBool>,
}

impl Reporter for WatchFor<'_> {
    fn on_hit(&mut self, hit: &Hit) {
        if hit.score == 20 {
            self.found.store(true, Ordering::SeqCst);
        }
        self.inner.on_hit(hit);
    }
    fn on_speed(&mut self, total: f64, per_device: &[f64]) {
        self.inner.on_speed(total, per_device);
    }
}
