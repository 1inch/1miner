//! secp256k1 vanity search on OpenCL.
//!
//! Ported from profanity2's dispatcher. The offset arithmetic that makes a hit
//! usable, and the result-slot layout it is read out of, live in
//! [`crate::profanity`] and are shared with the Metal backend; what is here is
//! how OpenCL enumerates candidates.
//!
//! Three large scratch buffers hold the batched-inversion state, sized
//! `inverse_size * inverse_multiple` elements of 32 bytes each.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miner_core::{ModeConfig, ProfanityConfig, ScoreFn};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_uchar, cl_uint};

use super::{DeviceId, build_program, cl_err, enumerate_devices};
use crate::profanity::{
    MpNumber, MpPoint, ResultSlot, RoundContext, be_bytes_to_ulong4, check_offset_fields,
    device_seed, precomp_table,
};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter, combine};
use crate::{
    Backend, BackendError, DeviceInfo, EXACT_CAPACITY, Job, KeccakVariant, MAX_SCORE, Progress,
    RESULT_SLOTS, Reporter, Result, kernels,
};

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
    let precomp = precomp_table();

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
    precomp: &[MpPoint],
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
        Buffer::<MpPoint>::create(
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
        Buffer::<ResultSlot>::create(
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

    let mut results = vec![ResultSlot::default(); RESULT_SLOTS];
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
    let zeros = vec![ResultSlot::default(); RESULT_SLOTS];

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
            device_index: info.index,
            seed: &seed,
            round,
        };
        let found = if job.is_exact() {
            // Slot 0 counts this round's matches, including any the buffer had
            // no room for.
            context.drain_exact(&results, results[0].found)
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
    use super::*;

    /// The exact kernel has no score and no bar, so it is picked by
    /// `--exact` rather than by the scoring function, which is left at
    /// whatever the mode parsed.
    #[test]
    fn exact_selects_its_own_kernel_whatever_the_scorer_says() {
        for function in [ScoreFn::Benchmark, ScoreFn::Leading, ScoreFn::Mirror] {
            assert_eq!(
                score_kernel_name(function),
                match function {
                    ScoreFn::Benchmark => "profanity_iterate_score_benchmark",
                    ScoreFn::Leading => "profanity_iterate_score_leading",
                    _ => "profanity_iterate_score_mirror",
                }
            );
        }
    }

    /// Every name this picks has to exist in the source, and a typo here
    /// fails at kernel creation on a device rather than in the suite.
    #[test]
    fn every_iterate_kernel_name_is_declared_in_the_source() {
        let names = [
            ScoreFn::Benchmark,
            ScoreFn::ZeroBytes,
            ScoreFn::Matching,
            ScoreFn::Leading,
            ScoreFn::Range,
            ScoreFn::Mirror,
            ScoreFn::Doubles,
            ScoreFn::LeadingRange,
        ]
        .map(score_kernel_name);

        for name in names {
            let macro_call = name.replace("profanity_iterate_score_", "");
            assert!(
                kernels::PROFANITY.contains(&format!("PROFANITY_SCORE_KERNEL({macro_call})")),
                "profanity.cl does not instantiate {name}"
            );
        }
        assert!(
            kernels::PROFANITY.contains("__kernel void profanity_iterate_exact_match"),
            "profanity.cl does not declare the exact kernel"
        );
    }
}
