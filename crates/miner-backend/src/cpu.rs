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
    MineMode, ModeConfig, ProfanityConfig, SaltConfig,
    scoring::score,
    secp256k1::{Point, generator, point_add},
};

use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{Backend, DeviceInfo, Hit, Job, Reporter, Result};

pub struct CpuBackend {
    infos: Vec<DeviceInfo>,
    threads: usize,
}

impl CpuBackend {
    pub fn new(threads: Option<usize>) -> Self {
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
            }],
            threads,
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
        let best = AtomicU64::new(job.initial_threshold() as u64);
        let hits: Mutex<Vec<Hit>> = Mutex::new(Vec::new());
        let start = Instant::now();
        let mut reported = 0usize;

        std::thread::scope(|scope| {
            for shard in 0..self.threads {
                let (counter, best, hits) = (&counter, &best, &hits);
                let threads = self.threads;
                scope.spawn(move || {
                    let mut round: u32 = 0;
                    loop {
                        if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
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
                                let value = score(&job.score, &addresses[lane]) as u64;
                                if value == 0 || value <= best.load(Ordering::Relaxed) {
                                    continue;
                                }
                                // --exact leaves the bar pinned, so every full
                                // match is reported rather than only the first.
                                if !job.is_exact() {
                                    best.store(value, Ordering::Relaxed);
                                }
                                let salt = cfg.salt_at(0, pair[lane], round);
                                let magic = (cfg.mode == MineMode::Nft).then(|| {
                                    let mut m = [0u8; 16];
                                    m.copy_from_slice(&salt[..16]);
                                    m
                                });
                                hits.lock().unwrap().push(Hit {
                                    score: value as u32,
                                    address: addresses[lane],
                                    salt: Some(salt),
                                    magic,
                                    offset: None,
                                    device_index: 0,
                                    // Derived on the CPU to begin with, so
                                    // there is nothing left to cross-check.
                                    verified: true,
                                });
                            }

                            done += lanes as u64;
                            gid = gid.wrapping_add(stride * 2);
                            if done % 4096 < lanes as u64 {
                                counter.fetch_add(4096, Ordering::Relaxed);
                                if should_stop()
                                    || job.duration.is_some_and(|d| start.elapsed() >= d)
                                {
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

        drain(&hits, reported, reporter);
        Ok(())
    }

    /// Walk the seed public key forward one generator step at a time. Each step
    /// costs a modular inversion, so this is orders of magnitude slower than
    /// the GPU path and exists to verify it rather than to compete with it.
    fn run_profanity(
        &self,
        cfg: &ProfanityConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let counter = AtomicU64::new(0);
        let hits: Mutex<Vec<Hit>> = Mutex::new(Vec::new());
        let start = Instant::now();
        let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);

        let g = generator();
        let mut point: Point = cfg.seed_public_key;
        let mut offset: u64 = 0;
        let mut best = job.initial_threshold();

        while !should_stop() && job.duration.is_none_or(|d| start.elapsed() < d) {
            let Some(next) = point_add(Some(&point), Some(&g)) else {
                break;
            };
            point = next;
            offset += 1;

            let address = cfg.address_for_point(&point);
            let value = score(&job.score, &address);
            if value > best {
                if !job.is_exact() {
                    best = value;
                }
                let mut bytes = [0u8; 32];
                bytes[24..].copy_from_slice(&offset.to_be_bytes());
                hits.lock().unwrap().push(Hit {
                    score: value,
                    address,
                    salt: None,
                    magic: None,
                    offset: Some(bytes),
                    device_index: 0,
                    verified: true,
                });
            }

            counter.fetch_add(1, Ordering::Relaxed);
            if offset % 512 == 0 {
                let reported = hits.lock().unwrap().len();
                drain(&hits, reported.saturating_sub(1), reporter);
                meter.sample(counter.load(Ordering::Relaxed));
                let rate = meter.rate();
                reporter.on_speed(rate, &[rate]);
            }
        }

        if let Some(summary) = meter.summary() {
            reporter.on_summary(&summary);
        }
        Ok(())
    }
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
    hits: &Mutex<Vec<Hit>>,
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
        reported = drain(hits, reported, reporter);

        meter.sample(counter.load(Ordering::Relaxed));
        let rate = meter.rate();
        reporter.on_speed(rate, &[rate]);

        if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
            // The meter lives here, so the summary has to be reported here too.
            if let Some(summary) = meter.summary() {
                reporter.on_summary(&summary);
            }
            return reported;
        }
    }
}

fn drain(hits: &Mutex<Vec<Hit>>, from: usize, reporter: &mut dyn Reporter) -> usize {
    let guard = hits.lock().unwrap();
    for hit in guard.iter().skip(from) {
        reporter.on_hit(hit);
    }
    guard.len()
}
