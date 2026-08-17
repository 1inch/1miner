//! Salt search on OpenCL, covering create2, create3 and 1nft.
//!
//! One thread per GPU, each with its own context and queue. Within a thread the
//! loop mirrors ERADICATE2's dispatcher: read the previous round's results
//! without blocking, queue the next round behind that read, and wait only on
//! the read. The queue is in-order, so a kernel is always in flight.

use std::sync::atomic::Ordering;
use std::time::Instant;

use miner_core::{ModeConfig, SaltConfig};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_uchar, cl_uint};

use super::{
    DeviceId, DeviceRun, buffer, build_program, cl_err, enumerate_devices, run_devices,
    state_define,
};
use crate::salt::{SaltRound, SaltSlot};
use crate::{
    Backend, BackendError, DeviceInfo, EXACT_CAPACITY, Job, KeccakVariant, MAX_SCORE, RESULT_SLOTS,
    Reporter, Result, chunks, kernels, queue_best, wire,
};

/// The scoring mode and the exact masks are different shapes bound to the same
/// kernel argument, so both reach the device as plain bytes.
fn bytes_of<T>(values: &[T]) -> &[u8] {
    // SAFETY: `T` here is only ever a `#[repr(C)]` plain-data struct with no
    // padding that matters, the slice is borrowed for the call, and u8 has an
    // alignment of 1, so the reinterpreted slice is in bounds and aligned.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
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

    run_devices(
        ids,
        infos,
        job,
        reporter,
        should_stop,
        |device_id, info, run, start| {
            run_device(
                device_id,
                info,
                cfg,
                job,
                &source,
                &options,
                run,
                should_stop,
                start,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn run_device(
    device_id: DeviceId,
    info: &DeviceInfo,
    cfg: &SaltConfig,
    job: &Job,
    source: &str,
    options: &str,
    run: DeviceRun<'_>,
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

    let mut result_buf: Buffer<SaltSlot> =
        buffer(&context, CL_MEM_READ_WRITE, RESULT_SLOTS, "result")?;

    // One mode for scoring, or the masks for --exact. Both are read-only and
    // uploaded once, so they share a buffer slot in the kernel's arguments.
    let patterns = wire::patterns(job.exact.as_deref());
    let modes = [wire::Mode::from(&job.score)];
    let (mode_bytes, mode_len) = if job.is_exact() {
        (bytes_of(&patterns), size_of_val(patterns.as_slice()))
    } else {
        (bytes_of(&modes), size_of::<wire::Mode>())
    };

    let mut mode_buf: Buffer<u8> = buffer(&context, CL_MEM_READ_ONLY, mode_len, "mode")?;

    let mut results = vec![SaltSlot::default(); RESULT_SLOTS];
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

    let reader = SaltRound {
        cfg,
        job,
        device_index: info.index,
    };
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
    let zeros = vec![SaltSlot::default(); RESULT_SLOTS];

    loop {
        if should_stop() || job.expired(start) {
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
        for (offset, this) in chunks(round_size, chunk) {
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
        }
        queue.flush().map_err(cl_err("flush failed"))?;
        read_event.wait().map_err(cl_err("result read failed"))?;

        run.counter.fetch_add(round_size as u64, Ordering::Relaxed);

        if job.is_exact() {
            // Slot 0 counts this round's matches, including any the buffer had
            // no room for.
            let found = reader.drain_exact(&results, results[0].found);
            if !found.is_empty() {
                run.hits.lock().unwrap().extend(found);
            }
        } else {
            // The bar is read, raised and queued against in one step, for the
            // reason `queue_best` gives: devices share it, and separating the
            // three lets a later round report a score below an earlier one.
            queue_best(run.hits, run.best_score, |threshold| {
                reader.take_best(&results, threshold)
            });
        }
        // After the drain rather than before, because reporting publishes the
        // score: one load then raises this kernel's bar to the best any device
        // has found, instead of leaving a device that is behind writing results
        // the host reads and throws away. In --exact mode the shared value is
        // pinned, so this is a no-op.
        local_best = run.best_score.load(Ordering::Relaxed) as cl_uchar;
    }

    queue.finish().map_err(cl_err("finish failed"))?;
    Ok(())
}
