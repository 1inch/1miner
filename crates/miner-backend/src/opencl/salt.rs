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
    Backend, BackendError, DeviceInfo, Hit, Job, KeccakVariant, Reporter, Result, kernels,
};

/// Highest score the result buffer has a slot for.
pub const MAX_SCORE: usize = 40;

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
        "-D SALT_MAX_SCORE={MAX_SCORE} -D SALT_INITHASH={}",
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
    let hits: Mutex<Vec<Hit>> = Mutex::new(Vec::new());
    // Shared so a strong hit on one GPU raises the bar on all of them. In
    // --exact mode the bar is pinned instead and never moves, so every full
    // match gets reported rather than only improvements.
    let best_score = AtomicU64::new(job.initial_threshold() as u64);
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

fn drain_hits(hits: &Mutex<Vec<Hit>>, from: usize, reporter: &mut dyn Reporter) -> usize {
    let guard = hits.lock().unwrap();
    for hit in guard.iter().skip(from) {
        reporter.on_hit(hit);
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
    hits: &Mutex<Vec<Hit>>,
    best_score: &AtomicU64,
    should_stop: &(dyn Fn() -> bool + Sync),
    start: Instant,
) -> Result<()> {
    let device = Device::new(device_id.0);
    let context = Context::from_device(&device).map_err(cl_err("failed to create context"))?;
    let queue = CommandQueue::create_default(&context, 0)
        .map_err(cl_err("failed to create command queue"))?;

    let program = build_program(&context, &device, source, options, !job.tuning.no_cache)?;
    let kernel =
        Kernel::create(&program, "salt_iterate").map_err(cl_err("missing salt_iterate"))?;

    // SAFETY: a null host pointer with neither CL_MEM_USE_HOST_PTR nor
    // CL_MEM_COPY_HOST_PTR set asks OpenCL to own the allocation, so there is no
    // host memory whose lifetime has to be upheld here.
    let mut result_buf = unsafe {
        Buffer::<ClResult>::create(
            &context,
            CL_MEM_READ_WRITE,
            MAX_SCORE + 1,
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate result buffer"))?;
    // SAFETY: as above, OpenCL owns the allocation.
    let mut mode_buf =
        unsafe { Buffer::<ClMode>::create(&context, CL_MEM_READ_ONLY, 1, std::ptr::null_mut()) }
            .map_err(cl_err("failed to allocate mode buffer"))?;

    let mut results = vec![ClResult::default(); MAX_SCORE + 1];
    let modes = [ClMode::from(&job.score)];
    // SAFETY: both writes are blocking, so the source slices only have to be
    // live for the duration of the call, and each holds exactly as many elements
    // as the buffer it fills was created with.
    unsafe {
        queue
            .enqueue_write_buffer(&mut result_buf, CL_BLOCKING, 0, &results, &[])
            .map_err(cl_err("failed to clear result buffer"))?;
        queue
            .enqueue_write_buffer(&mut mode_buf, CL_BLOCKING, 0, &modes, &[])
            .map_err(cl_err("failed to upload mode"))?;
    }

    let device_index = info.index as cl_uint;
    let round_size = job.tuning.round_size;
    let chunk = job.tuning.work_max.unwrap_or(round_size).max(1);
    let local = job.tuning.work_size;
    let mut round: cl_uint = 0;
    let mut local_best: cl_uchar = job.initial_threshold() as cl_uchar;
    // In --exact mode the kernel's one-slot-per-score guard would suppress
    // every match after the first, so the slot is cleared each round.
    let zeros = vec![ClResult::default(); MAX_SCORE + 1];

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
            // SAFETY: the argument types and their order match `salt_iterate`'s
            // parameters in kernels/opencl/salt.cl, and every one of them —
            // both buffers and the three scalars — outlives the enqueue, which
            // is drained by `queue.flush()` and the event wait below.
            unsafe {
                let mut exec = ExecuteKernel::new(&kernel);
                exec.set_arg(&result_buf)
                    .set_arg(&mode_buf)
                    .set_arg(&local_best)
                    .set_arg(&device_index)
                    .set_arg(&round)
                    .set_global_work_offset(offset)
                    .set_global_work_size(this);
                if local > 0 {
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

        if let Some(hit) = take_best(&results, cfg, job, info, best_score) {
            if !job.is_exact() {
                local_best = hit.score as cl_uchar;
            }
            hits.lock().unwrap().push(hit);
        }
    }

    queue.finish().map_err(cl_err("finish failed"))?;
    Ok(())
}

/// Result slots are indexed by score, so the best hit is the highest occupied
/// slot above what has already been reported.
fn take_best(
    results: &[ClResult],
    cfg: &SaltConfig,
    job: &Job,
    info: &DeviceInfo,
    best_score: &AtomicU64,
) -> Option<Hit> {
    let threshold = best_score.load(Ordering::Relaxed);
    for score in (1..=MAX_SCORE).rev() {
        if results[score].found == 0 {
            continue;
        }
        if score as u64 <= threshold {
            return None;
        }
        // --exact keeps the bar where it is so later full matches still report.
        if !job.is_exact() {
            best_score.store(score as u64, Ordering::Relaxed);
        }

        let mut salt = [0u8; 32];
        salt.copy_from_slice(&results[score].salt);
        let mut address = [0u8; 20];
        address.copy_from_slice(&results[score].hash);

        // The kernel rebuilds the salt separately from the hashing path, so
        // re-deriving here is what proves the two still agree.
        let verified = !job.verify || cfg.address_for_salt(&salt) == address;

        let magic = (cfg.mode == MineMode::Nft).then(|| {
            let mut m = [0u8; 16];
            m.copy_from_slice(&salt[..16]);
            m
        });

        return Some(Hit {
            score: score as u32,
            address,
            salt: Some(salt),
            magic,
            offset: None,
            device_index: info.index,
            verified,
        });
    }
    None
}
