//! CPU backend.
//!
//! Portable and always available, so it runs anywhere and needs no driver. Its
//! main job is to be the reference the accelerated backends are checked
//! against, but it is a usable miner for small searches and it is what makes
//! `--backend cpu` a sensible fallback on a machine with no GPU.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miner_core::{
    Address, ModeConfig, ProfanityConfig, SaltConfig,
    scoring::{first_exact_match, score},
    secp256k1::{Point, add_scalars_mod_n, generator, point_add, scalar_mul_generator},
};
use rand::RngCore;

use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{Backend, BackendError, DeviceInfo, Hit, Job, Progress, Reporter, Result, drain_hits};

pub struct CpuBackend {
    infos: Vec<DeviceInfo>,
    threads: usize,
    profanity_base: Option<[u8; 32]>,
}

impl CpuBackend {
    pub fn new(threads: Option<usize>) -> Self {
        Self::build(threads, None)
    }

    /// Start the profanity walk at a chosen offset rather than a random one.
    ///
    /// This exists for the agreement test: pointed at an offset the OpenCL
    /// kernel has already reported, the walk derives the address for that exact
    /// scalar through entirely separate code, so the two can be compared. There
    /// is no flag for it, because a real search wants the random start.
    pub fn starting_at(threads: Option<usize>, offset: [u8; 32]) -> Self {
        Self::build(threads, Some(offset))
    }

    fn build(threads: Option<usize>, profanity_base: Option<[u8; 32]>) -> Self {
        let threads = threads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        Self {
            infos: vec![DeviceInfo {
                index: 0,
                name: format!("CPU ({threads} threads)"),
                compute_units: threads as u32,
                global_memory: 0,
                driver: None,
            }],
            threads,
            profanity_base,
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn devices(&self) -> &[DeviceInfo] {
        &self.infos
    }

    fn run(
        &mut self,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        match &job.mode {
            ModeConfig::Salt(cfg) => self.run_salt(cfg, job, reporter, should_stop),
            ModeConfig::Profanity(cfg) => self.run_profanity(cfg, job, reporter, should_stop),
        }
    }
}

impl CpuBackend {
    fn run_salt(
        &self,
        cfg: &SaltConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let counter = AtomicU64::new(0);
        let best = AtomicU64::new(0);
        let hits: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
        let start = Instant::now();
        let mut reported = 0usize;

        std::thread::scope(|scope| {
            for shard in 0..self.threads {
                let (counter, best, hits) = (&counter, &best, &hits);
                let threads = self.threads;
                scope.spawn(move || {
                    let mut round: u32 = 0;
                    loop {
                        if should_stop() || job.expired(start) {
                            break;
                        }
                        round = round.wrapping_add(1);

                        let mut done: u64 = 0;
                        let mut gid = shard as u32;
                        let stride = threads as u32;
                        while (gid as usize) < job.tuning.round_size {
                            // Two candidates at a time, so the aarch64 path can
                            // use a single two-lane Keccak permutation. The
                            // partner is this shard's next work item, and it is
                            // dropped when the round ends between the two.
                            let partner = gid.wrapping_add(stride);
                            let paired = (partner as usize) < job.tuning.round_size;
                            let pair = [gid, if paired { partner } else { gid }];
                            let addresses = derive_pair(cfg, pair, round);
                            let lanes = if paired { 2 } else { 1 };

                            for lane in 0..lanes {
                                let Some((value, pattern)) = examine(job, &addresses[lane], best)
                                else {
                                    continue;
                                };
                                let salt = cfg.salt_at(0, pair[lane], round);
                                hits.lock().unwrap().push(Progress::Hit(Hit {
                                    score: value,
                                    address: addresses[lane],
                                    salt: Some(salt),
                                    magic: cfg.magic_of(&salt),
                                    offset: None,
                                    pattern,
                                    device_index: 0,
                                    // Derived on the CPU to begin with, so
                                    // there is nothing left to cross-check.
                                    verified: true,
                                }));
                            }

                            done += lanes as u64;
                            gid = gid.wrapping_add(stride * 2);
                            if done % 4096 < lanes as u64 {
                                counter.fetch_add(4096, Ordering::Relaxed);
                                if should_stop() || job.expired(start) {
                                    break;
                                }
                            }
                        }
                        counter.fetch_add(done % 4096, Ordering::Relaxed);
                    }
                });
            }

            reported = poll(&hits, reported, &counter, start, job, reporter, should_stop);
        });

        drain_hits(&hits, reported, reporter);
        Ok(())
    }

    /// Walk forward from a random point on the seed key's line, one generator
    /// step at a time. Each step costs a modular inversion, so this is orders
    /// of magnitude slower than the GPU path and exists to verify it rather
    /// than to compete with it.
    ///
    /// The start is drawn per run for the reason the OpenCL path draws one per
    /// device: from a fixed start, a second run against the same public key
    /// re-covers the offsets the first one already did, so it adds nothing to a
    /// search and two people working from the same published key find the same
    /// addresses.
    fn run_profanity(
        &self,
        cfg: &ProfanityConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let counter = AtomicU64::new(0);
        let start = Instant::now();
        let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);

        let g = generator();
        let base = self.profanity_base.unwrap_or_else(random_base);
        // A zero base leaves `scalar_mul_generator` at infinity, and adding
        // that is the identity, so the walk simply starts at the seed itself.
        let mut point: Point = point_add(
            Some(&cfg.seed_public_key),
            scalar_mul_generator(&base).as_ref(),
        )
        .ok_or_else(|| {
            BackendError::Other("the starting offset cancels the seed public key".into())
        })?;
        let mut steps: u64 = 0;
        let best = AtomicU64::new(0);

        while !should_stop() && !job.expired(start) {
            let address = cfg.address_for_point(&point);
            if let Some((value, pattern)) = examine(job, &address, &best) {
                let offset = offset_scalar(&base, steps);
                // The offset is rebuilt from the base and the step count rather
                // than read off the walk, so the two can drift; re-deriving the
                // address from the offset alone is what would notice.
                let verified = !job.verify || cfg.address_for_offset(&offset) == Some(address);
                // Straight out, rather than into a collection for the speed
                // poll to drain the way run_salt needs: this loop is
                // single-threaded, and draining every 512 steps kept only
                // whichever hit was newest.
                reporter.on_hit(&Hit {
                    score: value,
                    address,
                    salt: None,
                    magic: None,
                    offset: Some(offset),
                    pattern,
                    device_index: 0,
                    verified,
                });
            }

            counter.fetch_add(1, Ordering::Relaxed);
            steps += 1;
            if steps % 512 == 0 {
                meter.sample(counter.load(Ordering::Relaxed));
                let rate = meter.rate();
                reporter.on_speed(rate, &[rate]);
            }

            let Some(next) = point_add(Some(&point), Some(&g)) else {
                break;
            };
            point = next;
        }

        if let Some(summary) = meter.summary() {
            reporter.on_summary(&summary);
        }
        Ok(())
    }
}

/// Where this run's walk starts: 240 random bits.
///
/// The top two bytes stay clear so that `seed_priv + offset` cannot carry past
/// 256 bits, which is what the OpenCL path reserves the top of its own offset
/// for. Cryptographic quality is not needed — the security of the result rests
/// on the user's seed key, which never enters this process — but `rand::rng()`
/// is OS-seeded, unlike a clock read.
/// Is this address worth reporting, and if so with what score and which mask?
///
/// The two questions the backends ask, in one place because the CPU asks both
/// of them in two loops. `--exact` wants every address satisfying a mask and
/// has no bar; scoring wants each improvement on the best seen anywhere, and
/// raises the bar as it goes.
fn examine(job: &Job, address: &Address, best: &AtomicU64) -> Option<(u32, Option<usize>)> {
    match job.exact.as_deref() {
        Some(masks) => {
            let pattern = first_exact_match(masks, address)?;
            Some((masks[pattern].constrained_bytes(), Some(pattern)))
        }
        None => {
            let value = score(&job.score, address) as u64;
            if value == 0 || value <= best.load(Ordering::Relaxed) {
                return None;
            }
            best.store(value, Ordering::Relaxed);
            Some((value as u32, None))
        }
    }
}

fn random_base() -> [u8; 32] {
    let mut base = [0u8; 32];
    rand::rng().fill_bytes(&mut base);
    base[0] = 0;
    base[1] = 0;
    base
}

/// The offset naming the point `steps` generator steps beyond `base`.
///
/// `base` is below 2²⁴⁰ and `steps` below 2⁶⁴, so the sum is below 2²⁴¹ and the
/// reduction mod n never fires — which is what keeps this exact, since
/// `add_scalars_mod_n` reduces only one multiple of n.
fn offset_scalar(base: &[u8; 32], steps: u64) -> [u8; 32] {
    let mut delta = [0u8; 32];
    delta[24..].copy_from_slice(&steps.to_be_bytes());
    add_scalars_mod_n(base, &delta)
}

/// Whether the two-lane NEON permutation is used, cached because this is read
/// in the hot loop.
///
/// `MINER_NO_NEON=1` forces the plain reference. That exists both to A/B the two
/// and as a safety valve: if the SIMD path ever misbehaves on some hardware,
/// there is a way to keep mining while it is investigated.
#[cfg(target_arch = "aarch64")]
fn neon_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("MINER_NO_NEON").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Derive two candidate addresses, using the two-lane NEON permutation on
/// aarch64 and the plain reference elsewhere. Both paths are checked against
/// each other in tests, so this stays a performance choice rather than a
/// behavioural one.
#[inline]
fn derive_pair(cfg: &SaltConfig, gids: [u32; 2], round: u32) -> [miner_core::Address; 2] {
    #[cfg(target_arch = "aarch64")]
    if neon_enabled() {
        return crate::neon::addresses(cfg, 0, gids, round);
    }

    [
        cfg.address_for_salt(&cfg.salt_at(0, gids[0], round)),
        cfg.address_for_salt(&cfg.salt_at(0, gids[1], round)),
    ]
}

#[allow(clippy::too_many_arguments)]
fn poll(
    hits: &Mutex<Vec<Progress>>,
    mut reported: usize,
    counter: &AtomicU64,
    start: Instant,
    job: &Job,
    reporter: &mut dyn Reporter,
    should_stop: &(dyn Fn() -> bool + Sync),
) -> usize {
    let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);
    loop {
        std::thread::sleep(Duration::from_millis(250));
        reported = drain_hits(hits, reported, reporter);

        meter.sample(counter.load(Ordering::Relaxed));
        let rate = meter.rate();
        reporter.on_speed(rate, &[rate]);

        if should_stop() || job.expired(start) {
            // The meter lives here, so the summary has to be reported here too.
            if let Some(summary) = meter.summary() {
                reporter.on_summary(&summary);
            }
            return reported;
        }
    }
}
