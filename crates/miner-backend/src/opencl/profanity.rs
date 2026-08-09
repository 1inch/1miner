//! secp256k1 vanity search on OpenCL.
//!
//! Ported from profanity2's dispatcher. Each work item starts at
//! `seed_pub + (seed + (id << 192)) * G` and every round advances all points by
//! one generator step, so the scalar for a hit is `seed + round + (id << 192)`.
//! That offset is what gets reported: added to the user's seed private key it
//! yields the private key for the found address, and the miner never sees a
//! private key at any point.
//!
//! Three large scratch buffers hold the batched-inversion state, sized
//! `inverse_size * inverse_multiple` elements of 32 bytes each.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miner_core::{ModeConfig, ProfanityConfig, ScoreFn, secp256k1::generator_table};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_uchar, cl_uint};

use super::{DeviceId, build_program, cl_err, enumerate_devices};
use crate::{
    Backend, BackendError, DeviceInfo, Hit, Job, KeccakVariant, Reporter, Result, kernels,
};

pub const MAX_SCORE: usize = 40;

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

fn iterate_kernel_name(function: ScoreFn) -> &'static str {
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

/// A random starting offset. Only the top 16 bits are cleared, so adding this
/// to the user's seed private key cannot overflow 256 bits. Cryptographic
/// quality is not needed: the security of the result comes from the user's own
/// seed key, this only keeps devices from covering the same ground.
fn random_seed() -> ClUlong4 {
    let mut bytes = [0u8; 32];
    getrandom(&mut bytes);
    let mut lanes = be_bytes_to_ulong4(&bytes);
    lanes.0[3] &= 0x0000_ffff_ffff_ffff;
    lanes
}

fn getrandom(buf: &mut [u8; 32]) {
    // Avoid a dependency here; the caller-visible randomness requirement is
    // only that two devices do not repeat each other's work.
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut state = (nanos as u64) ^ (std::process::id() as u64).rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    for chunk in buf.chunks_mut(8) {
        // SplitMix64.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_be_bytes()[..chunk.len()]);
    }
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
    let source = program_source(job.keccak);
    let options = format!(
        "-D PROFANITY_INVERSE_SIZE={} -D PROFANITY_MAX_SCORE={MAX_SCORE}",
        job.tuning.inverse_size
    );

    // Shared by every device and identical for all of them.
    let table = generator_table();
    let precomp: Vec<ClPoint> = table
        .iter()
        .map(|p| {
            let (x, y) = p.to_bytes();
            ClPoint { x: to_mp(&x), y: to_mp(&y) }
        })
        .collect();

    let counters: Vec<AtomicU64> = ids.iter().map(|_| AtomicU64::new(0)).collect();
    let hits: Mutex<Vec<Hit>> = Mutex::new(Vec::new());
    let best_score = AtomicU64::new(job.initial_threshold() as u64);
    let failure: Mutex<Option<BackendError>> = Mutex::new(None);
    let start = Instant::now();
    let mut reported = 0usize;

    std::thread::scope(|scope| {
        for (slot, (device_id, info)) in ids.iter().zip(infos).enumerate() {
            let (source, options, precomp) = (&source, &options, &precomp);
            let (counters, hits, best_score, failure) =
                (&counters, &hits, &best_score, &failure);

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

            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                let per_device: Vec<f64> = counters
                    .iter()
                    .map(|c| c.load(Ordering::Relaxed) as f64 / elapsed)
                    .collect();
                reporter.on_speed(per_device.iter().sum(), &per_device);
            }

            let expired = job.duration.is_some_and(|d| start.elapsed() >= d);
            if expired || should_stop() || failure.lock().unwrap().is_some() {
                break;
            }
        }
    });

    drain_hits(&hits, reported, reporter);
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
    cfg: &ProfanityConfig,
    job: &Job,
    source: &str,
    options: &str,
    precomp: &[ClPoint],
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
    let kernel_init =
        Kernel::create(&program, "profanity_init").map_err(cl_err("missing profanity_init"))?;
    let kernel_inverse = Kernel::create(&program, "profanity_inverse")
        .map_err(cl_err("missing profanity_inverse"))?;
    let iterate_name = iterate_kernel_name(job.score.function);
    let kernel_iterate =
        Kernel::create(&program, iterate_name).map_err(cl_err("missing iterate kernel"))?;

    let size = job.tuning.profanity_round_size();

    let mut mem_precomp = unsafe {
        Buffer::<ClPoint>::create(&context, CL_MEM_READ_ONLY, precomp.len(), std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate precomp buffer"))?;
    let mem_delta_x = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate deltaX buffer"))?;
    let mem_inversed = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate inverse buffer"))?;
    let mem_prev_lambda = unsafe {
        Buffer::<MpNumber>::create(&context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate lambda buffer"))?;
    let mut mem_result = unsafe {
        Buffer::<ClResult>::create(&context, CL_MEM_READ_WRITE, MAX_SCORE + 1, std::ptr::null_mut())
    }
    .map_err(cl_err("failed to allocate result buffer"))?;
    let mut mem_data1 =
        unsafe { Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 20, std::ptr::null_mut()) }
            .map_err(cl_err("failed to allocate data1"))?;
    let mut mem_data2 =
        unsafe { Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 20, std::ptr::null_mut()) }
            .map_err(cl_err("failed to allocate data2"))?;

    let mut results = vec![ClResult::default(); MAX_SCORE + 1];
    unsafe {
        queue
            .enqueue_write_buffer(&mut mem_precomp, CL_BLOCKING, 0, precomp, &[])
            .map_err(cl_err("failed to upload precomp table"))?;
        queue
            .enqueue_write_buffer(&mut mem_result, CL_BLOCKING, 0, &results, &[])
            .map_err(cl_err("failed to clear results"))?;
        queue
            .enqueue_write_buffer(&mut mem_data1, CL_BLOCKING, 0, &job.score.data1, &[])
            .map_err(cl_err("failed to upload data1"))?;
        queue
            .enqueue_write_buffer(&mut mem_data2, CL_BLOCKING, 0, &job.score.data2, &[])
            .map_err(cl_err("failed to upload data2"))?;
    }

    let (seed_x_bytes, seed_y_bytes) = cfg.seed_public_key.to_bytes();
    let seed = random_seed();
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
    let mut local_best: cl_uchar = job.initial_threshold() as cl_uchar;
    // --exact needs the per-score slot cleared so repeat full matches record.
    let zeros = vec![ClResult::default(); MAX_SCORE + 1];

    loop {
        if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
            break;
        }

        let read_event =
            unsafe { queue.enqueue_read_buffer(&mem_result, CL_NON_BLOCKING, 0, &mut results, &[]) }
                .map_err(cl_err("failed to read results"))?;

        // In-order queue: this lands after the read above and before the
        // kernels below.
        if job.is_exact() {
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
            |exec| unsafe {
                exec.set_arg(&mem_delta_x).set_arg(&mem_inversed);
            },
        )?;
        enqueue_chunked(&queue, &kernel_iterate, size, work_max, local, |exec| unsafe {
            exec.set_arg(&mem_delta_x)
                .set_arg(&mem_inversed)
                .set_arg(&mem_prev_lambda)
                .set_arg(&mem_result)
                .set_arg(&mem_data1)
                .set_arg(&mem_data2)
                .set_arg(&local_best)
                .set_arg(&is_contract);
        })?;

        queue.flush().map_err(cl_err("flush failed"))?;
        read_event.wait().map_err(cl_err("result read failed"))?;

        round += 1;
        counter.fetch_add(size as u64, Ordering::Relaxed);

        let threshold = best_score.load(Ordering::Relaxed);
        for score in (1..=MAX_SCORE).rev() {
            if results[score].found == 0 {
                continue;
            }
            if score as u64 <= threshold {
                break;
            }
            if !job.is_exact() {
                best_score.store(score as u64, Ordering::Relaxed);
                local_best = score as cl_uchar;
            }

            let mut address = [0u8; 20];
            address.copy_from_slice(&results[score].found_hash);
            let offset = offset_scalar(&seed, round, results[score].found_id);

            // Walk the seed public key forward by the reported offset and check
            // it lands on the reported address. A mismatch means the offset
            // accounting is wrong and the key would be useless.
            let verified = !job.verify
                || miner_core::secp256k1::address_for_offset(&cfg.seed_public_key, &offset)
                    .map(|p| if cfg.contract { miner_core::create_address(&p, 0) } else { p })
                    == Some(address);

            hits.lock().unwrap().push(Hit {
                score: score as u32,
                address,
                salt: None,
                magic: None,
                offset: Some(offset),
                device_index: info.index,
                verified,
            });
            break;
        }
    }

    queue.finish().map_err(cl_err("finish failed"))?;
    Ok(())
}

/// Split a launch into `work_max` sized pieces, as the reference dispatcher does.
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
        unsafe {
            let mut exec = ExecuteKernel::new(kernel);
            set_args(&mut exec);
            exec.set_global_work_offset(offset).set_global_work_size(run);
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
        for _ in 0..16 {
            assert_eq!(random_seed().0[3] >> 48, 0);
        }
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
