//! secp256k1 vanity search on OpenCL.
//!
//! Ported from profanity2's dispatcher. Each work item starts at
//! `seed_pub + (seed + (id << 192)) * G` and every round advances all points by
//! one generator step, so the scalar for a hit is `seed + round + (id << 192)`.
//! That offset is what gets reported: added to the user's seed private key it
//! yields the private key for the found address, and the miner never sees a
//! private key at any point.
//!
//! Devices are partitioned inside that offset rather than left to chance: the
//! top lane of `seed` carries a device slot above the bits the kernel adds `id`
//! into, so two devices cannot walk the same sequence however they are drawn.
//!
//! Three large scratch buffers hold the batched-inversion state, sized
//! `inverse_size * inverse_multiple` elements of 32 bytes each.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miner_core::{ModeConfig, ProfanityConfig, ScoreFn, ScoreSpec, secp256k1::generator_table};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_uchar, cl_uint};
use rand::RngCore;

use super::{DeviceId, build_program, cl_err, enumerate_devices};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter, combine};
use crate::{
    Backend, BackendError, DeviceInfo, EXACT_CAPACITY, Hit, Job, KeccakVariant, MAX_SCORE,
    Progress, RESULT_SLOTS, Reporter, Result, kernels,
};

/// `mp_number` from profanity2's types.hpp: eight 32-bit words, 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct MpNumber {
    d: [cl_uint; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ClPoint {
    x: MpNumber,
    y: MpNumber,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ClResult {
    found: cl_uint,
    found_id: cl_uint,
    found_hash: [cl_uchar; 20],
}

/// OpenCL `ulong4`: four 64-bit lanes, 32-byte aligned.
#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
struct ClUlong4([u64; 4]);

pub struct ProfanityBackend {
    ids: Vec<DeviceId>,
    infos: Vec<DeviceInfo>,
}

impl ProfanityBackend {
    pub fn new(skip: &[usize]) -> Result<Self> {
        let found = enumerate_devices(skip)?;
        let (ids, infos) = found.into_iter().unzip();
        Ok(Self { ids, infos })
    }
}

impl Backend for ProfanityBackend {
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
        let ModeConfig::Profanity(cfg) = &job.mode else {
            return Err(BackendError::Unsupported("opencl profanity", "salt"));
        };
        run_profanity(&self.ids, &self.infos, cfg, job, reporter, should_stop)
    }
}

pub fn program_source(keccak: KeccakVariant) -> String {
    format!("{}\n{}", keccak.source(), kernels::PROFANITY)
}

/// Which iterate kernel a job runs. `--exact` asks a different question and so
/// runs a different kernel over a different result layout; see the comment on
/// `profanity_iterate_exact_match`.
fn iterate_kernel_name(job: &Job) -> &'static str {
    if job.is_exact() {
        return "profanity_iterate_exact_match";
    }
    score_kernel_name(job.score.function)
}

fn score_kernel_name(function: ScoreFn) -> &'static str {
    match function {
        ScoreFn::Benchmark => "profanity_iterate_score_benchmark",
        ScoreFn::ZeroBytes => "profanity_iterate_score_zerobytes",
        ScoreFn::Matching => "profanity_iterate_score_matching",
        ScoreFn::Leading => "profanity_iterate_score_leading",
        ScoreFn::Range => "profanity_iterate_score_range",
        ScoreFn::Mirror => "profanity_iterate_score_mirror",
        ScoreFn::Doubles => "profanity_iterate_score_doubles",
        ScoreFn::LeadingRange => "profanity_iterate_score_leadingrange",
    }
}

/// Big-endian 32 bytes into four 64-bit lanes, least significant lane first.
fn be_bytes_to_ulong4(bytes: &[u8; 32]) -> ClUlong4 {
    let mut lanes = [0u64; 4];
    for (i, lane) in lanes.iter_mut().enumerate() {
        let start = 24 - i * 8;
        *lane = u64::from_be_bytes(bytes[start..start + 8].try_into().unwrap());
    }
    ClUlong4(lanes)
}

/// The most significant lane of an offset is a packed field. From the top: 16
/// bits left clear so `seed_priv + offset` cannot overflow 256 bits, 16 bits of
/// device slot, and 32 bits the kernel adds the work-item id into.
const ID_BITS: u32 = 32;
const MAX_ROUND_SIZE: u64 = 1 << ID_BITS;
const MAX_DEVICES: u64 = 1 << 16;

/// One device's starting offset: 192 random bits, so two runs do not cover the
/// same ground, above a device slot no other device of this run can reach.
///
/// Cryptographic quality is not needed — the security of the result comes from
/// the user's own seed key, which never enters this process — but `rand::rng()`
/// is per-thread and OS-seeded, unlike a clock read, and it is what the salt
/// modes already use.
fn device_seed(device_index: usize) -> ClUlong4 {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let mut lanes = be_bytes_to_ulong4(&bytes);
    lanes.0[3] = (device_index as u64) << ID_BITS;
    lanes
}

/// Both fields have to hold for the separation to mean anything: a round wider
/// than its id field would reach into the next device's slot, and a slot above
/// its own field into the bits that must stay clear. Neither limit is anywhere
/// near a tuning that fits in memory — the default round is 2²² work items —
/// but checking them is what makes the separation structural rather than
/// assumed.
fn check_offset_fields(round_size: usize, infos: &[DeviceInfo]) -> Result<()> {
    if round_size as u64 > MAX_ROUND_SIZE {
        return Err(BackendError::Other(format!(
            "--inverse-size x --inverse-multiple is {round_size} work items, \
             above the {MAX_ROUND_SIZE} one round can address"
        )));
    }
    let highest = infos.iter().map(|i| i.index).max().unwrap_or(0) as u64;
    if highest >= MAX_DEVICES {
        return Err(BackendError::Other(format!(
            "device index {highest} is above the {MAX_DEVICES} an offset can keep apart"
        )));
    }
    Ok(())
}

/// `seed + round + (found_id << 192)` as a big-endian 32-byte scalar.
///
/// profanity2 open-codes this with a shortcut carry that misfires when a lane
/// is already zero; a full 256-bit add is used here instead.
fn offset_scalar(seed: &ClUlong4, round: u64, found_id: u32) -> [u8; 32] {
    let mut lanes = seed.0;
    let mut carry = round as u128;
    for lane in lanes.iter_mut() {
        let sum = *lane as u128 + (carry & 0xFFFF_FFFF_FFFF_FFFF);
        *lane = sum as u64;
        carry = (carry >> 64) + (sum >> 64);
    }
    lanes[3] = lanes[3].wrapping_add(found_id as u64);

    let mut out = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        let start = 24 - i * 8;
        out[start..start + 8].copy_from_slice(&lane.to_be_bytes());
    }
    out
}

fn run_profanity(
    ids: &[DeviceId],
    infos: &[DeviceInfo],
    cfg: &ProfanityConfig,
    job: &Job,
    reporter: &mut dyn Reporter,
    should_stop: &(dyn Fn() -> bool + Sync),
) -> Result<()> {
    check_offset_fields(job.tuning.profanity_round_size(), infos)?;

    let source = program_source(job.keccak);
    let options = format!(
        "-D PROFANITY_INVERSE_SIZE={} -D PROFANITY_MAX_SCORE={MAX_SCORE} \
         -D PROFANITY_EXACT_CAPACITY={EXACT_CAPACITY}",
        job.tuning.inverse_size
    );

    // Shared by every device and identical for all of them.
    let table = generator_table();
    let precomp: Vec<ClPoint> = table
        .iter()
        .map(|p| {
            let (x, y) = p.to_bytes();
            ClPoint {
                x: to_mp(&x),
                y: to_mp(&y),
            }
        })
        .collect();

    let counters: Vec<AtomicU64> = ids.iter().map(|_| AtomicU64::new(0)).collect();
    let hits: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
    // The exact path has no bar and leaves this at zero.
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
            let (source, options, precomp) = (&source, &options, &precomp);
            let (counters, hits, best_score, failure) = (&counters, &hits, &best_score, &failure);

            scope.spawn(move || {
                let outcome = run_device(
                    *device_id,
                    info,
                    cfg,
                    job,
                    source,
                    options,
                    precomp,
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

fn to_mp(be: &[u8; 32]) -> MpNumber {
    let mut d = [0u32; 8];
    for (i, word) in d.iter_mut().enumerate() {
        let start = 28 - i * 4;
        *word = u32::from_be_bytes(be[start..start + 4].try_into().unwrap());
    }
    MpNumber { d }
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
    cfg: &ProfanityConfig,
    job: &Job,
    source: &str,
    options: &str,
    precomp: &[ClPoint],
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
    let kernel_init =
        Kernel::create(&program, "profanity_init").map_err(cl_err("missing profanity_init"))?;
    let kernel_inverse = Kernel::create(&program, "profanity_inverse")
        .map_err(cl_err("missing profanity_inverse"))?;
    let iterate_name = iterate_kernel_name(job);
    let kernel_iterate =
        Kernel::create(&program, iterate_name).map_err(cl_err("missing iterate kernel"))?;

    // Scoring reads one mask from each; the exact kernel reads `patterns.len()`
    // of them laid end to end.
    let patterns = job.exact.as_deref().unwrap_or_default();
    let (data1, data2): (Vec<cl_uchar>, Vec<cl_uchar>) = if job.is_exact() {
        (
            patterns.iter().flat_map(|p| p.data1).collect(),
            patterns.iter().flat_map(|p| p.data2).collect(),
        )
    } else {
        (job.score.data1.to_vec(), job.score.data2.to_vec())
    };

    let size = job.tuning.profanity_round_size();

    // SAFETY: every allocation below passes a null host pointer with neither
    // CL_MEM_USE_HOST_PTR nor CL_MEM_COPY_HOST_PTR, so OpenCL owns the storage
    // and no host lifetime has to be upheld. This applies to all seven.
    let mut mem_precomp = unsafe {
        Buffer::<ClPoint>::create(
            &context,
            CL_MEM_READ_ONLY,
            precomp.len(),
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate precomp buffer"))?;
    // SAFETY: as above.
    let mem_delta_x = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate deltaX buffer"))?;
    // SAFETY: as above.
    let mem_inversed = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate inverse buffer"))?;
    // SAFETY: as above.
    let mem_prev_lambda = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate lambda buffer"))?;
    // SAFETY: as above.
    let mut mem_result = unsafe {
        Buffer::<ClResult>::create(
            &context,
            CL_MEM_READ_WRITE,
            RESULT_SLOTS,
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate result buffer"))?;
    // SAFETY: as above.
    let mut mem_data1 = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            data1.len(),
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate data1"))?;
    // SAFETY: as above.
    let mut mem_data2 = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            data2.len(),
            std::ptr::null_mut(),
        )
    }
    .map_err(cl_err("failed to allocate data2"))?;

    let mut results = vec![ClResult::default(); RESULT_SLOTS];
    // SAFETY: all four writes are blocking, so each source only has to be live
    // for the duration of its call, and each holds exactly as many elements as
    // the buffer it fills was created with.
    unsafe {
        queue
            .enqueue_write_buffer(&mut mem_precomp, CL_BLOCKING, 0, precomp, &[])
            .map_err(cl_err("failed to upload precomp table"))?;
        queue
            .enqueue_write_buffer(&mut mem_result, CL_BLOCKING, 0, &results, &[])
            .map_err(cl_err("failed to clear results"))?;
        queue
            .enqueue_write_buffer(&mut mem_data1, CL_BLOCKING, 0, &data1, &[])
            .map_err(cl_err("failed to upload data1"))?;
        queue
            .enqueue_write_buffer(&mut mem_data2, CL_BLOCKING, 0, &data2, &[])
            .map_err(cl_err("failed to upload data2"))?;
    }

    let (seed_x_bytes, seed_y_bytes) = cfg.seed_public_key.to_bytes();
    let seed = device_seed(info.index);
    let seed_x = be_bytes_to_ulong4(&seed_x_bytes);
    let seed_y = be_bytes_to_ulong4(&seed_y_bytes);
    let is_contract: cl_uchar = cfg.contract.into();
    let local = job.tuning.work_size;
    let work_max = job.tuning.work_max.unwrap_or(size).max(1);

    // Seeding touches every element of the three scratch buffers, so it is
    // chunked to keep individual enqueues small.
    let init_chunk = (size / 20).clamp(1, work_max);
    let mut initialized = 0usize;
    while initialized < size {
        let run = init_chunk.min(size - initialized);
        // SAFETY: the argument types and their order match `profanity_init`'s
        // parameters in kernels/opencl/profanity.cl, and every buffer and scalar
        // passed outlives the enqueue, which is drained by the flush below and
        // the `queue.finish()` after the loop.
        unsafe {
            ExecuteKernel::new(&kernel_init)
                .set_arg(&mem_precomp)
                .set_arg(&mem_delta_x)
                .set_arg(&mem_prev_lambda)
                .set_arg(&mem_result)
                .set_arg(&seed)
                .set_arg(&seed_x)
                .set_arg(&seed_y)
                .set_global_work_offset(initialized)
                .set_global_work_size(run)
                .enqueue_nd_range(&queue)
                .map_err(cl_err("profanity_init failed"))?;
        }
        queue.flush().map_err(cl_err("flush failed"))?;
        initialized += run;
    }
    queue.finish().map_err(cl_err("initialization failed"))?;

    let mut round: u64 = 0;
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
            queue.enqueue_read_buffer(&mem_result, CL_NON_BLOCKING, 0, &mut results, &[])
        }
        .map_err(cl_err("failed to read results"))?;

        // In-order queue: this lands after the read above and before the
        // kernels below.
        if job.is_exact() {
            // SAFETY: non-blocking, so the source has to outlive the transfer.
            // `zeros` is allocated before the loop and dropped after it, and
            // nothing writes to it.
            unsafe {
                queue
                    .enqueue_write_buffer(&mut mem_result, CL_NON_BLOCKING, 0, &zeros, &[])
                    .map_err(cl_err("failed to reset result buffer"))?;
            }
        }

        enqueue_chunked(
            &queue,
            &kernel_inverse,
            size / job.tuning.inverse_size,
            work_max,
            local,
            // SAFETY: both arguments match `profanity_inverse`'s parameters in
            // kernels/opencl/profanity.cl, and both buffers are captured by
            // reference from this function's scope, so they outlive every
            // enqueue the closure feeds.
            |exec| unsafe {
                exec.set_arg(&mem_delta_x).set_arg(&mem_inversed);
            },
        )?;
        enqueue_chunked(
            &queue,
            &kernel_iterate,
            size,
            work_max,
            local,
            // SAFETY: the arguments and their order match the iterate kernel
            // selected by `iterate_kernel_name`, and every buffer and scalar is
            // captured by reference from this function's scope, so all of them
            // outlive the enqueues the closure feeds. The seventh differs
            // between the two kernels and is chosen the same way the name was.
            |exec| unsafe {
                exec.set_arg(&mem_delta_x)
                    .set_arg(&mem_inversed)
                    .set_arg(&mem_prev_lambda)
                    .set_arg(&mem_result)
                    .set_arg(&mem_data1)
                    .set_arg(&mem_data2);
                if job.is_exact() {
                    exec.set_arg(&pattern_count);
                } else {
                    exec.set_arg(&local_best);
                }
                exec.set_arg(&is_contract);
            },
        )?;

        queue.flush().map_err(cl_err("flush failed"))?;
        read_event.wait().map_err(cl_err("result read failed"))?;

        round += 1;
        counter.fetch_add(size as u64, Ordering::Relaxed);

        let context = RoundContext {
            cfg,
            job,
            info,
            seed: &seed,
            round,
        };
        let found = if job.is_exact() {
            context.drain_exact(&results)
        } else {
            let threshold = best_score.load(Ordering::Relaxed);
            // Every device's progress, not just this one's, so a device that is
            // behind stops writing results the host reads and throws away.
            local_best = threshold as cl_uchar;
            match context.take_best(&results, threshold) {
                Some((score, hit)) => {
                    best_score.store(u64::from(score), Ordering::Relaxed);
                    local_best = score as cl_uchar;
                    vec![Progress::Hit(hit)]
                }
                None => Vec::new(),
            }
        };
        if !found.is_empty() {
            hits.lock().unwrap().extend(found);
        }
    }

    queue.finish().map_err(cl_err("finish failed"))?;
    Ok(())
}

/// What turning a result slot into a `Hit` needs beyond the slot itself.
struct RoundContext<'a> {
    cfg: &'a ProfanityConfig,
    job: &'a Job,
    info: &'a DeviceInfo,
    seed: &'a ClUlong4,
    round: u64,
}

impl RoundContext<'_> {
    /// Build the hit a filled result slot describes.
    ///
    /// The offset is rebuilt from the seed, the round and the work-item id
    /// rather than read off the kernel, so walking the seed public key forward
    /// by it and comparing is what catches offset accounting that has gone
    /// wrong — which would otherwise hand over a key controlling a different
    /// address.
    fn hit_from(&self, slot: &ClResult, score: u32, pattern: Option<usize>) -> Hit {
        let mut address = [0u8; 20];
        address.copy_from_slice(&slot.found_hash);
        let offset = offset_scalar(self.seed, self.round, slot.found_id);

        Hit {
            score,
            address,
            salt: None,
            magic: None,
            offset: Some(offset),
            pattern,
            device_index: self.info.index,
            verified: !self.job.verify || self.cfg.address_for_offset(&offset) == Some(address),
        }
    }

    /// Result slots are indexed by score, so the best hit is the highest
    /// occupied slot above what has already been reported.
    fn take_best(&self, results: &[ClResult], threshold: u64) -> Option<(u32, Hit)> {
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

    /// Slots are the round's matches in arrival order, with slot 0 counting
    /// them all — including any the buffer had no room for.
    fn drain_exact(&self, results: &[ClResult]) -> Vec<Progress> {
        let total = results[0].found;
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
                device_index: self.info.index,
            });
        }
        found
    }
}

/// Split a launch into `work_max` sized pieces, as the reference dispatcher does.
///
/// `set_args` is where the kernel's arguments are bound, so matching them to the
/// kernel's parameters is the caller's responsibility; each call site carries the
/// safety comment for its own argument list.
fn enqueue_chunked(
    queue: &CommandQueue,
    kernel: &Kernel,
    total: usize,
    work_max: usize,
    local: usize,
    set_args: impl Fn(&mut ExecuteKernel),
) -> Result<()> {
    let mut offset = 0usize;
    while offset < total {
        let run = work_max.min(total - offset);
        // SAFETY: binding the arguments belongs to `set_args`, and each call
        // site states its own case; what this block adds is the work-item range,
        // where `offset + run` never exceeds `total`.
        unsafe {
            let mut exec = ExecuteKernel::new(kernel);
            set_args(&mut exec);
            exec.set_global_work_offset(offset)
                .set_global_work_size(run);
            if local > 0 && run % local == 0 {
                exec.set_local_work_size(local);
            }
            exec.enqueue_nd_range(queue)
                .map_err(cl_err("kernel enqueue failed"))?;
        }
        offset += run;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn info(index: usize) -> DeviceInfo {
        DeviceInfo {
            index,
            name: String::new(),
            compute_units: 0,
            global_memory: 0,
        }
    }

    #[test]
    fn offset_is_seed_plus_round_plus_shifted_id() {
        let seed = ClUlong4([5, 0, 0, 0]);
        let offset = offset_scalar(&seed, 7, 0);
        assert_eq!(u64::from_be_bytes(offset[24..].try_into().unwrap()), 12);

        // found_id lands in the most significant lane, i.e. shifted by 192 bits.
        let with_id = offset_scalar(&seed, 0, 3);
        assert_eq!(u64::from_be_bytes(with_id[..8].try_into().unwrap()), 3);
        assert_eq!(u64::from_be_bytes(with_id[24..].try_into().unwrap()), 5);
    }

    /// profanity2's shortcut carry treats an already-zero lane as a carry out.
    /// A full add must not.
    #[test]
    fn carry_only_propagates_on_real_overflow() {
        let seed = ClUlong4([u64::MAX, 0, 0, 0]);
        let offset = offset_scalar(&seed, 1, 0);
        assert_eq!(u64::from_be_bytes(offset[24..].try_into().unwrap()), 0);
        assert_eq!(u64::from_be_bytes(offset[16..24].try_into().unwrap()), 1);

        let no_carry = ClUlong4([1, 0, 0, 0]);
        let offset = offset_scalar(&no_carry, 1, 0);
        assert_eq!(u64::from_be_bytes(offset[16..24].try_into().unwrap()), 0);
    }

    #[test]
    fn seed_clears_the_top_bits_so_a_sum_cannot_overflow() {
        for device in [0, 1, 7, MAX_DEVICES as usize - 1] {
            assert_eq!(device_seed(device).0[3] >> 48, 0);
        }
    }

    #[test]
    fn seed_reserves_the_top_lane_for_the_device_slot() {
        for device in [0, 1, 7, MAX_DEVICES as usize - 1] {
            assert_eq!(device_seed(device).0[3], (device as u64) << ID_BITS);
        }
    }

    /// The whole point of the packing: the widest permitted round on one device
    /// stops short of the next device's slot, so no work item of one device can
    /// land on an offset another device reaches.
    #[test]
    fn the_widest_round_stops_short_of_the_next_device_slot() {
        // The largest id a permitted round produces, as the kernel reports it.
        let widest = u32::try_from(MAX_ROUND_SIZE - 1).expect("a round must fit the uint foundId");
        // Identical low lanes, as if the RNG had failed both devices, and the
        // highest round against the lowest: only the top lane can separate them.
        let shared = |device: u64| ClUlong4([9, 9, 9, device << ID_BITS]);

        for device in 0..4 {
            let last = offset_scalar(&shared(device), u64::MAX >> 1, widest);
            let first_of_next = offset_scalar(&shared(device + 1), 0, 0);
            assert!(
                last < first_of_next,
                "device {device} reaches into the next slot"
            );
        }
    }

    /// The predecessor derived all 256 bits from a clock read, so two device
    /// threads starting together usually drew the same seed.
    #[test]
    fn the_random_part_of_a_seed_differs_every_draw() {
        let drawn: HashSet<[u64; 4]> = (0..1000).map(|_| device_seed(0).0).collect();
        assert_eq!(drawn.len(), 1000);
    }

    #[test]
    fn a_round_or_a_rig_too_large_for_the_offset_fields_is_rejected() {
        assert!(check_offset_fields(MAX_ROUND_SIZE as usize, &[info(0)]).is_ok());
        assert!(check_offset_fields(MAX_ROUND_SIZE as usize + 1, &[info(0)]).is_err());

        let highest = MAX_DEVICES as usize - 1;
        assert!(check_offset_fields(1, &[info(0), info(highest)]).is_ok());
        assert!(check_offset_fields(1, &[info(0), info(highest + 1)]).is_err());
    }

    #[test]
    fn public_key_lanes_are_least_significant_first() {
        let mut be = [0u8; 32];
        be[31] = 1;
        assert_eq!(be_bytes_to_ulong4(&be).0, [1, 0, 0, 0]);
        let mut be = [0u8; 32];
        be[0] = 1;
        assert_eq!(be_bytes_to_ulong4(&be).0, [0, 0, 0, 1 << 56]);
    }
}
