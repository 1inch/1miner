//! `1miner self-test`: prove the selected backend derives the same addresses as
//! the CPU reference before a long run is trusted to it.
//!
//! The interesting check for the salt modes is the planted target. A work item
//! is chosen, its address computed on the CPU, and a full 20-byte mask built
//! from that address. The backend then has to find exactly that work item and
//! report the salt for it. Passing exercises the pre-image construction, the
//! keccak padding, the second CREATE hash, the scoring, and the separate salt
//! reconstruction path inside the kernel, all at once.
//!
//! profanity cannot be planted that way — a work item's address is not
//! addressable in advance the way `salt_at` makes a salt one — so it is checked
//! by the property that protects a real run: the offset reported for a hit has
//! to name the address reported with it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use miner_backend::{Hit, Job, Reporter, Tuning};
use miner_core::{
    DEFAULT_PROXY_CODE_HASH, MineMode, ModeConfig, PROXY_CHILD_BYTECODE, ProfanityConfig,
    SaltConfig, ScoreSpec, create3_address, keccak256, parse_address, secp256k1::generator,
};

use crate::cli::CommonArgs;

/// Small enough to finish instantly, large enough to span many work groups.
const ROUND: usize = 1 << 16;
const TARGET_GID: u32 = 12_345;
const ROUND_INDEX: u32 = 1;

/// Enough profanity hits to be worth checking. The run ends as soon as they
/// arrive rather than burning the timeout, which on a GPU is a round or two.
const PROFANITY_HITS: usize = 3;

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
        let name = format!("{} planted target", mode.as_str());
        report(&mut failures, &name, planted_target(mode, common));
    }

    for contract in [false, true] {
        let name = if contract {
            "profanity --contract offset agreement"
        } else {
            "profanity offset agreement"
        };
        if common.backend == "metal" {
            // open_backend rejects the mode here, and a FAIL would be saying
            // this device is untrustworthy when the mode simply is not built.
            println!("  skip  {name} (metal has no secp256k1 kernel)");
            continue;
        }
        report(&mut failures, name, profanity_agreement(contract, common));
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

/// A check that could not run counts as a failure: the point of the self-test
/// is to say whether this device can be trusted, and a run that never happened
/// answers that no more than a wrong address does.
fn report(failures: &mut usize, name: &str, outcome: anyhow::Result<bool>) {
    match outcome {
        Ok(ok) => check(failures, name, ok),
        Err(e) => {
            check(failures, name, false);
            println!("      {e}");
        }
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
        mode: ModeConfig::Salt(cfg),
        score,
        keccak: common.keccak(),
        tuning: tuning.clone(),
        // The planted item is reached in round 1; allow a little slack.
        duration: Some(std::time::Duration::from_secs(20)),
        verify: true,
        exact: None,
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

    backend.run(
        &job,
        &mut WatchFor {
            inner: &mut collector,
            found: Arc::clone(&found),
        },
        &should_stop,
    )?;

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

/// Mine profanity for real, briefly, and re-derive every hit from the offset
/// reported with it.
///
/// This is the same check `--verify` makes during a run, which is a good one —
/// but it only fires on a hit, and on a long search that is hours in. Asking
/// for it here puts the answer before the rental rather than after it.
fn profanity_agreement(contract: bool, common: &CommonArgs) -> anyhow::Result<bool> {
    // The generator as the seed public key: its private half is 1, so anything
    // that fails here can be reproduced by hand.
    let cfg = ProfanityConfig {
        seed_public_key: generator(),
        contract,
    };

    let tuning = Tuning {
        work_size: common.work.min(64),
        // 255 x 64 is 16320 points, against the default 4.2M and its ~400 MB.
        // Enough to fill many work groups, brief enough to initialise at once.
        inverse_size: 255,
        inverse_multiple: 64,
        skip_devices: common.skip.clone(),
        no_cache: true,
        ..Tuning::default()
    };

    let job = Job {
        mode: ModeConfig::Profanity(cfg.clone()),
        // A loose bar, so hits arrive in the first round or two.
        score: ScoreSpec::zeros(),
        keccak: common.keccak(),
        tuning: tuning.clone(),
        duration: Some(Duration::from_secs(20)),
        verify: true,
        exact: None,
    };

    let mut backend = crate::open_backend(common, &job.mode, &tuning)?;
    let mut collector = Collector::default();
    let enough = Arc::new(AtomicBool::new(false));
    let should_stop = {
        let enough = Arc::clone(&enough);
        move || enough.load(Ordering::SeqCst)
    };

    backend.run(
        &job,
        &mut StopAfter {
            inner: &mut collector,
            enough: Arc::clone(&enough),
            wanted: PROFANITY_HITS,
        },
        &should_stop,
    )?;

    if collector.hits.is_empty() {
        anyhow::bail!("no hits in 20 seconds, so nothing was checked");
    }
    Ok(collector.hits.iter().all(|hit| {
        hit.verified
            && hit
                .offset
                .is_some_and(|offset| cfg.address_for_offset(&offset) == Some(hit.address))
    }))
}

/// Flip a flag once enough hits have arrived so the run can end early.
struct StopAfter<'a> {
    inner: &'a mut Collector,
    enough: Arc<AtomicBool>,
    wanted: usize,
}

impl Reporter for StopAfter<'_> {
    fn on_hit(&mut self, hit: &Hit) {
        self.inner.on_hit(hit);
        if self.inner.hits.len() >= self.wanted {
            self.enough.store(true, Ordering::SeqCst);
        }
    }
    fn on_speed(&mut self, total: f64, per_device: &[f64]) {
        self.inner.on_speed(total, per_device);
    }
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
