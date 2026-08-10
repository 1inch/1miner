//! Salt search on OpenCL, covering create2, create3 and 1nft.
//!
//! One thread per GPU, each with its own context and queue. Within a thread the
//! loop mirrors ERADICATE2's dispatcher: read the previous round's results
//! without blocking, queue the next round behind that read, and wait only on
//! the read. The queue is in-order, so a kernel is always in flight.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miner_core::{MineMode, ModeConfig, SaltConfig, ScoreSpec};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_uchar, cl_uint};

use super::{DeviceId, build_program, cl_err, enumerate_devices, state_define};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter, combine};
use crate::{
    Backend, BackendError, DeviceInfo, EXACT_CAPACITY, Hit, Job, KeccakVariant, MAX_SCORE,
    Progress, RESULT_SLOTS, Reporter, Result, kernels,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct ClMode {
    function: cl_uint,
    data1: [cl_uchar; 20],
    data2: [cl_uchar; 20],
}

impl From<&ScoreSpec> for ClMode {
    fn from(spec: &ScoreSpec) -> Self {
        Self {
            function: spec.function as cl_uint,
            data1: spec.data1,
            data2: spec.data2,
        }
    }
}

/// One `--exact` mask, matching `pattern` in kernels/opencl/salt.cl.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ClPattern {
    mask: [cl_uchar; 20],
    want: [cl_uchar; 20],
}

impl From<&ScoreSpec> for ClPattern {
    fn from(spec: &ScoreSpec) -> Self {
        Self {
            mask: spec.data1,
            want: spec.data2,
        }
    }
}

/// The scoring mode and the exact masks are different shapes bound to the same
/// kernel argument, so both reach the device as plain bytes.
fn bytes_of<T>(values: &[T]) -> &[u8] {
    // SAFETY: `T` here is only ever a `#[repr(C)]` plain-data struct with no
    // padding that matters, the slice is borrowed for the call, and u8 has an
    // alignment of 1, so the reinterpreted slice is in bounds and aligned.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct ClResult {
    salt: [cl_uchar; 32],
    hash: [cl_uchar; 20],
    found: cl_uint,
}

pub struct SaltBackend {
    ids: Vec<DeviceId>,
    infos: Vec<DeviceInfo>,
}

impl SaltBackend {
    pub fn new(skip: &[usize]) -> Result<Self> {
        let found = enumerate_devices(skip)?;
        let (ids, infos) = found.into_iter().unzip();
        Ok(Self { ids, infos })
    }
}

impl Backend for SaltBackend {
    fn name(&self) -> &'static str {
        "opencl"
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
        let ModeConfig::Salt(cfg) = &job.mode else {
            return Err(BackendError::Unsupported("opencl salt", "profanity"));
        };
        run_salt(&self.ids, &self.infos, cfg, job, reporter, should_stop)
    }
}

/// Compile-time defines for one salt job.
fn build_options(cfg: &SaltConfig) -> String {
    let mut options = format!(
        "-D SALT_MAX_SCORE={MAX_SCORE} -D SALT_EXACT_CAPACITY={EXACT_CAPACITY} \
         -D SALT_INITHASH={}",
        state_define(&cfg.state_words())
    );
    if cfg.mode.needs_second_hash() {
        options.push_str(" -D SALT_SECOND_HASH=1");
    }
    options
}

pub fn program_source(keccak: KeccakVariant) -> String {
    format!("{}\n{}", keccak.source(), kernels::SALT)
}

fn run_salt(
    ids: &[DeviceId],
    infos: &[DeviceInfo],
    cfg: &SaltConfig,
    job: &Job,
    reporter: &mut dyn Reporter,
    should_stop: &(dyn Fn() -> bool + Sync),
) -> Result<()> {
    let source = program_source(job.keccak);
    let options = build_options(cfg);

    let counters: Vec<AtomicU64> = ids.iter().map(|_| AtomicU64::new(0)).collect();
    let hits: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
    // Shared so a strong hit on one GPU raises the bar on all of them. The
    // exact path has no bar and leaves this at zero.
    let best_score = AtomicU64::new(0);
    let failure: Mutex<Option<BackendError>> = Mutex::new(None);
    let start = Instant::now();
    let mut reported = 0usize;
    let mut meters: Vec<SpeedMeter> = ids
        .iter()
        .map(|_| SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup))
        .collect();

    std::thread::scope(|scope| {
        for (slot, (device_id, info)) in ids.iter().zip(infos).enumerate() {
            let (source, options, counters) = (&source, &options, &counters);
            let (hits, best_score, failure) = (&hits, &best_score, &failure);

            scope.spawn(move || {
                let outcome = run_device(
                    *device_id,
                    info,
                    cfg,
                    job,
                    source,
                    options,
                    &counters[slot],
                    hits,
                    best_score,
                    should_stop,
                    start,
                );
                if let Err(e) = outcome {
                    let mut guard = failure.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(e);
                    }
                }
            });
        }

        // Poll for progress while the device threads work.
        loop {
            std::thread::sleep(Duration::from_millis(250));
            reported = drain_hits(&hits, reported, reporter);

            for (meter, counter) in meters.iter_mut().zip(counters.iter()) {
                meter.sample(counter.load(Ordering::Relaxed));
            }
            let per_device: Vec<f64> = meters.iter().map(SpeedMeter::rate).collect();
            reporter.on_speed(per_device.iter().sum(), &per_device);

            let expired = job.duration.is_some_and(|d| start.elapsed() >= d);
            if expired || should_stop() || failure.lock().unwrap().is_some() {
                break;
            }
        }
    });

    // Anything found between the last poll and shutdown.
    drain_hits(&hits, reported, reporter);
    let summaries: Vec<_> = meters.iter().map(SpeedMeter::summary).collect();
    if let Some(total) = combine(&summaries) {
        reporter.on_summary(&total);
    }

    match failure.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn drain_hits(hits: &Mutex<Vec<Progress>>, from: usize, reporter: &mut dyn Reporter) -> usize {
    let guard = hits.lock().unwrap();
    for found in guard.iter().skip(from) {
        match found {
            Progress::Hit(hit) => reporter.on_hit(hit),
            Progress::Dropped {
                count,
                device_index,
            } => reporter.on_dropped(*count, *device_index),
        }
    }
    guard.len()
}

#[allow(clippy::too_many_arguments)]
fn run_device(
    device_id: DeviceId,
    info: &DeviceInfo,
    cfg: &SaltConfig,
    job: &Job,
    source: &str,
    options: &str,
    counter: &AtomicU64,
    hits: &Mutex<Vec<Progress>>,
    best_score: &AtomicU64,
    should_stop: &(dyn Fn() -> bool + Sync),
    start: Instant,
) -> Result<()> {
    let device = Device::new(device_id.0);
    let context = Context::from_device(&device).map_err(cl_err("failed to create context"))?;
    let queue = CommandQueue::create_default(&context, 0)
        .map_err(cl_err("failed to create command queue"))?;

    let program = build_program(&context, &device, source, options, !job.tuning.no_cache)?;
    // --exact asks a different question and so runs a different kernel over a
    // different result layout; see the comment on `salt_iterate_exact`.
    let kernel_name = if job.is_exact() {
        "salt_iterate_exact"
    } else {
        "salt_iterate"
    };
    let kernel = Kernel::create(&program, kernel_name).map_err(cl_err("missing iterate kernel"))?;

    // SAFETY: a null host pointer with neither CL_MEM_USE_HOST_PTR nor
    // CL_MEM_COPY_HOST_PTR set asks OpenCL to own the allocation, so there is no
    // host memory whose lifetime has to be upheld here.
    let mut result_buf = unsafe {
        Buffer::<ClResult>::create(
            &context,
            CL_MEM_READ_WRITE,
            RESULT_SLOTS,
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate result buffer"))?;

    // One mode for scoring, or the masks for --exact. Both are read-only and
    // uploaded once, so they share a buffer slot in the kernel's arguments.
    let patterns: Vec<ClPattern> = job
        .exact
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(ClPattern::from)
        .collect();
    let modes = [ClMode::from(&job.score)];
    let (mode_bytes, mode_len) = if job.is_exact() {
        (bytes_of(&patterns), patterns.len() * size_of::<ClPattern>())
    } else {
        (bytes_of(&modes), size_of::<ClMode>())
    };

    // SAFETY: as above, OpenCL owns the allocation.
    let mut mode_buf =
        unsafe { Buffer::<u8>::create(&context, CL_MEM_READ_ONLY, mode_len, std::ptr::null_mut()) }
            .map_err(cl_err("failed to allocate mode buffer"))?;

    let mut results = vec![ClResult::default(); RESULT_SLOTS];
    // SAFETY: both writes are blocking, so the source slices only have to be
    // live for the duration of the call, and each holds exactly as many elements
    // as the buffer it fills was created with.
    unsafe {
        queue
            .enqueue_write_buffer(&mut result_buf, CL_BLOCKING, 0, &results, &[])
            .map_err(cl_err("failed to clear result buffer"))?;
        queue
            .enqueue_write_buffer(&mut mode_buf, CL_BLOCKING, 0, mode_bytes, &[])
            .map_err(cl_err("failed to upload mode"))?;
    }

    let device_index = info.index as cl_uint;
    let round_size = job.tuning.round_size;
    let chunk = job.tuning.work_max.unwrap_or(round_size).max(1);
    let local = job.tuning.work_size;
    let mut round: cl_uint = 0;
    let mut local_best: cl_uchar = 0;
    let pattern_count = patterns.len() as cl_uint;
    // The exact kernel appends from a counter in slot 0, so clearing it each
    // round is the protocol rather than a workaround: it is what lets the next
    // round start at slot 1 and what bounds the writes.
    let zeros = vec![ClResult::default(); RESULT_SLOTS];

    loop {
        if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
            break;
        }

        // SAFETY: the read is non-blocking, so `results` has to stay put and
        // untouched until the transfer completes. It is borrowed mutably by the
        // event, lives for the whole loop, and is only read after
        // `read_event.wait()` below.
        let read_event = unsafe {
            queue.enqueue_read_buffer(&result_buf, CL_NON_BLOCKING, 0, &mut results, &[])
        }
        .map_err(cl_err("failed to read results"))?;

        // The queue is in-order, so this lands after the read above has
        // captured the previous round and before this round's kernel runs.
        if job.is_exact() {
            // SAFETY: non-blocking, so the source has to outlive the transfer.
            // `zeros` is allocated before the loop and dropped after it, and
            // nothing writes to it.
            unsafe {
                queue
                    .enqueue_write_buffer(&mut result_buf, CL_NON_BLOCKING, 0, &zeros, &[])
                    .map_err(cl_err("failed to reset result buffer"))?;
            }
        }

        round = round.wrapping_add(1);
        let mut offset = 0usize;
        while offset < round_size {
            let this = chunk.min(round_size - offset);
            // SAFETY: the argument types and their order match the parameters
            // of whichever kernel `kernel_name` selected in
            // kernels/opencl/salt.cl — the third differs between them, and is
            // chosen alongside the name — and every one of them, both buffers
            // and the three scalars, outlives the enqueue, which is drained by
            // `queue.flush()` and the event wait below.
            unsafe {
                let mut exec = ExecuteKernel::new(&kernel);
                exec.set_arg(&result_buf).set_arg(&mode_buf);
                if job.is_exact() {
                    exec.set_arg(&pattern_count);
                } else {
                    exec.set_arg(&local_best);
                }
                exec.set_arg(&device_index)
                    .set_arg(&round)
                    .set_global_work_offset(offset)
                    .set_global_work_size(this);
                // A chunk the local size does not divide is rejected outright,
                // so a --size or --work-max that is not a multiple of --work
                // would abort the run instead of letting the driver choose.
                if local > 0 && this % local == 0 {
                    exec.set_local_work_size(local);
                }
                exec.enqueue_nd_range(&queue)
                    .map_err(cl_err("failed to enqueue salt_iterate"))?;
            }
            offset += this;
        }
        queue.flush().map_err(cl_err("flush failed"))?;
        read_event.wait().map_err(cl_err("result read failed"))?;

        counter.fetch_add(round_size as u64, Ordering::Relaxed);

        let found = if job.is_exact() {
            drain_exact(&results, cfg, job, info)
        } else {
            take_best(&results, cfg, job, info, best_score)
        };
        if !found.is_empty() {
            hits.lock().unwrap().extend(found);
        }
        // After take_best rather than before, because it stores when it
        // reports: one load then raises this kernel's bar to the best any
        // device has found, instead of leaving a device that is behind writing
        // results the host reads and throws away. In --exact mode the shared
        // value is pinned, so this is a no-op.
        local_best = best_score.load(Ordering::Relaxed) as cl_uchar;
    }

    queue.finish().map_err(cl_err("finish failed"))?;
    Ok(())
}

/// Build the hit a filled result slot describes.
///
/// The kernel rebuilds the salt separately from the hashing path, so
/// re-deriving the address from it here is what proves the two still agree.
fn hit_from(
    slot: &ClResult,
    score: u32,
    pattern: Option<usize>,
    cfg: &SaltConfig,
    job: &Job,
    info: &DeviceInfo,
) -> Hit {
    let mut salt = [0u8; 32];
    salt.copy_from_slice(&slot.salt);
    let mut address = [0u8; 20];
    address.copy_from_slice(&slot.hash);

    let magic = (cfg.mode == MineMode::Nft).then(|| {
        let mut m = [0u8; 16];
        m.copy_from_slice(&salt[..16]);
        m
    });

    Hit {
        score,
        address,
        salt: Some(salt),
        magic,
        offset: None,
        pattern,
        device_index: info.index,
        verified: !job.verify || cfg.address_for_salt(&salt) == address,
    }
}

/// Result slots are indexed by score, so the best hit is the highest occupied
/// slot above what has already been reported.
fn take_best(
    results: &[ClResult],
    cfg: &SaltConfig,
    job: &Job,
    info: &DeviceInfo,
    best_score: &AtomicU64,
) -> Vec<Progress> {
    let threshold = best_score.load(Ordering::Relaxed);
    for score in (1..=MAX_SCORE).rev() {
        if results[score].found == 0 {
            continue;
        }
        if score as u64 <= threshold {
            break;
        }
        best_score.store(score as u64, Ordering::Relaxed);
        return vec![Progress::Hit(hit_from(
            &results[score],
            score as u32,
            None,
            cfg,
            job,
            info,
        ))];
    }
    Vec::new()
}

/// Slots are the round's matches in arrival order, with slot 0 counting them
/// all — including any the buffer had no room for.
fn drain_exact(
    results: &[ClResult],
    cfg: &SaltConfig,
    job: &Job,
    info: &DeviceInfo,
) -> Vec<Progress> {
    let total = results[0].found;
    let stored = (total as usize).min(EXACT_CAPACITY);

    let masks = job.exact.as_deref().unwrap_or_default();
    let mut found: Vec<Progress> = (1..=stored)
        .map(|slot| {
            // `found` names the mask that matched in this layout, one-based so
            // that an untouched slot is distinguishable from mask 0.
            let pattern = results[slot].found.saturating_sub(1) as usize;
            let score = masks
                .get(pattern)
                .map_or(0, miner_core::ScoreSpec::constrained_bytes);
            Progress::Hit(hit_from(
                &results[slot],
                score,
                Some(pattern),
                cfg,
                job,
                info,
            ))
        })
        .collect();

    if let Some(dropped) = total.checked_sub(stored as u32).filter(|d| *d > 0) {
        found.push(Progress::Dropped {
            count: dropped,
            device_index: info.index,
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use miner_core::{DEFAULT_PROXY_CODE_HASH, parse_address};

    use super::*;
    use crate::Tuning;

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

    /// A slot holding a salt and the address it really derives, so the
    /// re-derivation the drains do has something true to agree with.
    fn filled(gid: u32, cfg: &SaltConfig) -> ClResult {
        let salt = cfg.salt_at(0, gid, 1);
        ClResult {
            salt,
            hash: cfg.address_for_salt(&salt),
            found: 1,
        }
    }

    /// One occupied slot, indexed by score as the scoring kernel writes it.
    fn results(score: usize, cfg: &SaltConfig) -> Vec<ClResult> {
        let mut slots = vec![ClResult::default(); RESULT_SLOTS];
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

    fn info() -> DeviceInfo {
        DeviceInfo {
            index: 0,
            name: String::new(),
            compute_units: 0,
            global_memory: 0,
        }
    }

    /// The round loop reloads its kernel's bar from the shared atomic, which
    /// only carries every device's progress because reporting publishes there.
    #[test]
    fn a_reported_hit_raises_the_shared_bar() {
        let cfg = config();
        let shared = AtomicU64::new(0);

        let found = take_best(&results(9, &cfg), &cfg, &job(&cfg, None), &info(), &shared);
        let hits = hits(&found);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 9);
        assert!(hits[0].verified);
        assert_eq!(shared.load(Ordering::Relaxed), 9);
    }

    /// A device another has already beaten reports nothing, so the bar it
    /// reloads afterwards is the leader's. That is the round where its kernel
    /// used to carry on writing results the host reads and throws away.
    #[test]
    fn a_beaten_hit_leaves_the_leader_in_the_shared_bar() {
        let cfg = config();
        let shared = AtomicU64::new(12);

        assert!(take_best(&results(9, &cfg), &cfg, &job(&cfg, None), &info(), &shared).is_empty());
        assert_eq!(shared.load(Ordering::Relaxed), 12);
    }

    /// Every match in the round comes back, not the one the buffer happened to
    /// keep first. This is the whole point of the exact path: the scoring
    /// layout has one slot per score, and in `--exact` every match scores the
    /// same, so all but one used to be unrecoverable.
    #[test]
    fn the_exact_drain_returns_every_match_in_the_round() {
        let cfg = config();
        let masks = vec![ScoreSpec::matching("00").unwrap()];

        let mut slots = vec![ClResult::default(); RESULT_SLOTS];
        for (i, gid) in (1..=5u32).enumerate() {
            slots[i + 1] = filled(gid, &cfg);
        }
        slots[0].found = 5;

        let found = drain_exact(&slots, &cfg, &job(&cfg, Some(masks)), &info());
        let hits = hits(&found);
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.verified));
        // Each slot carries its own work item's salt rather than a repeat of
        // one, which is what a drain reading the wrong index would produce.
        let salts: std::collections::HashSet<_> = hits.iter().map(|hit| hit.salt).collect();
        assert_eq!(salts.len(), 5);
        assert!(!found.iter().any(|p| matches!(p, Progress::Dropped { .. })));
    }

    /// Slot 0 counts every match, including the ones there was no room for, so
    /// a round that overflows can say by how much instead of quietly losing it.
    #[test]
    fn the_exact_drain_reports_what_would_not_fit() {
        let cfg = config();
        let masks = vec![ScoreSpec::matching("00").unwrap()];

        let mut slots = vec![ClResult::default(); RESULT_SLOTS];
        for (gid, slot) in slots
            .iter_mut()
            .enumerate()
            .take(EXACT_CAPACITY + 1)
            .skip(1)
        {
            *slot = filled(gid as u32, &cfg);
        }
        slots[0].found = EXACT_CAPACITY as u32 + 44;

        let found = drain_exact(&slots, &cfg, &job(&cfg, Some(masks)), &info());
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
        let masks = vec![
            ScoreSpec::matching("dead").unwrap(),
            ScoreSpec::matching("beefbeef").unwrap(),
        ];

        let mut slots = vec![ClResult::default(); RESULT_SLOTS];
        slots[1] = filled(1, &cfg);
        slots[1].found = 1;
        slots[2] = filled(2, &cfg);
        slots[2].found = 2;
        slots[0].found = 2;

        let found = drain_exact(&slots, &cfg, &job(&cfg, Some(masks)), &info());
        let hits = hits(&found);
        assert_eq!(hits[0].pattern, Some(0));
        assert_eq!(hits[1].pattern, Some(1));
        // The score is the mask's own constrained-byte count, so it stays
        // meaningful rather than repeating one number for every mask.
        assert_eq!((hits[0].score, hits[1].score), (2, 4));
    }
}
