//! secp256k1 vanity search on Metal.
//!
//! The offset arithmetic that makes a hit usable lives in [`crate::profanity`]
//! and is shared with the OpenCL backend; what is here is how Metal enumerates
//! candidates. Three scratch buffers hold the batched-inversion state, sized
//! `inverse_size * inverse_multiple` elements of 32 bytes each — about 400 MB
//! at the default tuning, which `-i` and `-I` are the knobs for.
//!
//! Each round is one command buffer with one encoder and two dispatches. A
//! compute encoder is serial unless asked otherwise, so the inverse pass is
//! ordered before, and visible to, the iterate pass without an explicit
//! barrier.

use std::time::Instant;

use miner_core::ProfanityConfig;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice,
    MTLResourceOptions,
};

use super::{
    CommandBuffer, MetalBackend, PIPELINE, SLOTS, clear, dispatch, encode, read_counter,
    read_slots, set_bytes, set_slice, threadgroup_width,
};
use crate::profanity::{
    MpNumber, MpPoint, ResultSlot, RoundContext, Ulong4, be_bytes_to_ulong4, check_offset_fields,
    device_seed, init_chunk, precomp_table,
};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{BackendError, EXACT_CAPACITY, Job, Progress, Reporter, Result, chunks, kernels, wire};

/// Matches `ProfParams` in kernels/metal/profanity.metal.
///
/// OpenCL passes the seed lanes as `ulong4` kernel arguments and the sizes as
/// `-D` defines. Metal has neither, so everything a job varies arrives here.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MtProfParams {
    seed: [u64; 4],
    seed_x: [u64; 4],
    seed_y: [u64; 4],
    /// Added to `thread_position_in_grid`. Metal has no equivalent of OpenCL's
    /// global work offset, so without this a launch split into chunks would
    /// start every chunk's ids at zero and the reported offsets would name
    /// addresses nobody found.
    id_base: u32,
    inverse_size: u32,
    score_max: u32,
    pattern_count: u32,
    exact_capacity: u32,
    contract: u32,
}

impl MetalBackend {
    pub(super) fn run_profanity(
        &self,
        cfg: &ProfanityConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let size = job.tuning.profanity_round_size();
        check_offset_fields(size, &self.infos)?;

        let device = &self.device;
        let queue = self.command_queue()?;

        let library = self.build_library(kernels::METAL_PROFANITY)?;
        let init = self.pipeline(&library, "profanity_init")?;
        let inverse = self.pipeline(&library, "profanity_inverse")?;
        // --exact asks a different question and so runs a different kernel over
        // a different result layout; see the comment on the exact kernel.
        let iterate = self.pipeline(
            &library,
            if job.is_exact() {
                "profanity_iterate_exact_match"
            } else {
                "profanity_iterate_score"
            },
        )?;

        // The three scratch buffers are never read by the CPU, so they stay in
        // private storage; the results and their counters have to come back.
        let scratch = |what: &'static str| {
            device
                .newBufferWithLength_options(
                    size * size_of::<MpNumber>(),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| {
                    BackendError::Other(format!(
                        "failed to allocate the {what} buffer: {size} work items need \
                         {} MB each of deltaX, inverse and lambda, so try a smaller \
                         --inverse-multiple",
                        size * size_of::<MpNumber>() / (1 << 20)
                    ))
                })
        };
        let delta_x = scratch("deltaX")?;
        let inversed = scratch("inverse")?;
        let prev_lambda = scratch("lambda")?;

        let precomp = precomp_table();
        let precomp_buffer = self.shared_buffer(precomp.len() * size_of::<MpPoint>(), "precomp")?;
        // SAFETY: the buffer was allocated with exactly this many bytes and is
        // in shared storage, so its contents pointer is writable by the CPU. No
        // GPU work has been submitted yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                precomp.as_ptr().cast::<u8>(),
                precomp_buffer.contents().as_ptr().cast::<u8>(),
                precomp.len() * size_of::<MpPoint>(),
            );
        }

        // Two of each, so a round can be committed while the one before it is
        // still being read. See the loop below.
        let mut results = Vec::with_capacity(PIPELINE);
        let mut flags = Vec::with_capacity(PIPELINE);
        for _ in 0..PIPELINE {
            let r = self.shared_buffer(SLOTS * size_of::<ResultSlot>(), "result")?;
            let f = self.shared_buffer(SLOTS * 4, "flag")?;
            clear(&r);
            clear(&f);
            results.push(r);
            flags.push(f);
        }

        let (seed_x, seed_y) = cfg.seed_public_key.to_bytes();
        let seed = device_seed(self.infos[0].index);
        let mode = wire::Mode::from(&job.score);
        let patterns = wire::patterns(job.exact.as_deref());

        let mut params = MtProfParams {
            seed: seed.0,
            seed_x: be_bytes_to_ulong4(&seed_x).0,
            seed_y: be_bytes_to_ulong4(&seed_y).0,
            id_base: 0,
            inverse_size: job.tuning.inverse_size as u32,
            score_max: 0,
            pattern_count: patterns.len() as u32,
            exact_capacity: EXACT_CAPACITY as u32,
            contract: cfg.contract.into(),
        };

        let work_max = job.tuning.work_max.unwrap_or(size).max(1);
        let start = Instant::now();

        // Seeding walks the precomp table with a full modular inversion per
        // point added, which is orders of magnitude more work per item than a
        // round. Left as one dispatch it can run long enough for the GPU
        // watchdog to take it for a hang, so it is chunked as the OpenCL path
        // chunks its enqueues.
        let init_group = threadgroup_width(&init, job.tuning.work_size);
        for (offset, run) in chunks(size, init_chunk(size, work_max)) {
            if should_stop() {
                return Ok(());
            }
            params.id_base = offset as u32;

            let (command_buffer, encoder) = encode(&queue)?;
            encoder.setComputePipelineState(&init);
            // SAFETY: the indices match `profanity_init`'s parameter positions
            // in kernels/metal/profanity.metal, and every buffer plus `params`
            // outlives the command buffer, which is waited on below.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&precomp_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&delta_x), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&prev_lambda), 0, 2);
                set_bytes(&encoder, &params, 3);
            }
            dispatch(&encoder, run, init_group);
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }

        let inverse_group = threadgroup_width(&inverse, job.tuning.work_size);
        let iterate_group = threadgroup_width(&iterate, job.tuning.work_size);
        let batches = size / job.tuning.inverse_size;

        let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);
        let rounds = Rounds {
            cfg,
            job,
            device_index: self.infos[0].index,
            seed: &seed,
        };
        let mut passes: u64 = 0;
        let mut best: u64 = 0;
        let mut hashes: u64 = 0;
        // The round the GPU is working on while the host reads the one before
        // it. Waiting on each round before submitting the next left the GPU
        // idle for the readback and the re-encode, measured at 0.22 ms a round
        // on an M4 Max: 2% at the default tuning, but 13% at `-I 1024`, since
        // the cost is per round rather than per candidate. A round is not much
        // work when the buffers have been made small to fit a smaller machine,
        // which is exactly when a user has already given up throughput.
        //
        // The rest of the per-round overhead is on the device and stays: the
        // two dispatches are serially dependent, as consecutive rounds are, so
        // nothing here can overlap the launches themselves.
        let mut in_flight: Option<(CommandBuffer, u64)> = None;

        loop {
            if should_stop() || job.expired(start) {
                break;
            }
            passes += 1;
            let slot = (passes as usize) % PIPELINE;

            // The exact kernel appends from the counter in foundFlags[0], so
            // clearing both buffers each round is the protocol rather than a
            // workaround: it is what lets the next round start at slot 1. The
            // scoring layout instead keeps its first-writer flags for the whole
            // run, since `score_max` only ever climbs past them.
            //
            // This pair was last written two rounds ago, and that round was
            // waited on while this one's predecessor was being encoded, so
            // clearing it here cannot race the GPU.
            if job.is_exact() {
                clear(&results[slot]);
                clear(&flags[slot]);
            }
            // Two rounds behind rather than one, since the round in flight has
            // not been read yet. The bar only suppresses writes the host would
            // discard anyway, so a stale one costs nothing but a few of them.
            params.score_max = best as u32;

            let (command_buffer, encoder) = encode(&queue)?;

            // A compute encoder created this way is serial, so the inverse
            // dispatches below are ordered before the iterate dispatches and
            // their writes are visible to them without an explicit barrier.
            encoder.setComputePipelineState(&inverse);
            for (offset, run) in chunks(batches, work_max) {
                params.id_base = offset as u32;
                // SAFETY: the indices match `profanity_inverse`'s parameter
                // positions in kernels/metal/profanity.metal; both buffers and
                // `params` outlive the command buffer, waited on below.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&delta_x), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&inversed), 0, 1);
                    set_bytes(&encoder, &params, 2);
                }
                dispatch(&encoder, run, inverse_group);
            }

            encoder.setComputePipelineState(&iterate);
            for (offset, run) in chunks(size, work_max) {
                params.id_base = offset as u32;
                // SAFETY: the indices match the iterate kernel selected above,
                // whose two forms differ only at index 4; every buffer and
                // `params` outlives the command buffer, waited on below.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&delta_x), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&inversed), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&prev_lambda), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&results[slot]), 0, 3);
                    if job.is_exact() {
                        set_slice(&encoder, &patterns, 4);
                    } else {
                        set_bytes(&encoder, &mode, 4);
                    }
                    set_bytes(&encoder, &params, 5);
                    encoder.setBuffer_offset_atIndex(Some(&flags[slot]), 0, 6);
                }
                dispatch(&encoder, run, iterate_group);
            }

            encoder.endEncoding();
            command_buffer.commit();

            // Read the previous round while this one runs. The rounds
            // themselves stay strictly ordered: each mutates deltaX and
            // prevLambda in place, so only the host side overlaps.
            if let Some((previous, done)) = in_flight.replace((command_buffer, passes)) {
                previous.waitUntilCompleted();
                let retired = (done as usize) % PIPELINE;
                rounds.retire(
                    &results[retired],
                    &flags[retired],
                    done,
                    &mut best,
                    reporter,
                );
                hashes += size as u64;
                meter.sample(hashes);
                let rate = meter.rate();
                reporter.on_speed(rate, &[rate]);
            }
        }

        if let Some((last, done)) = in_flight.take() {
            last.waitUntilCompleted();
            let retired = (done as usize) % PIPELINE;
            rounds.retire(
                &results[retired],
                &flags[retired],
                done,
                &mut best,
                reporter,
            );
            hashes += size as u64;
            meter.sample(hashes);
        }

        if let Some(summary) = meter.summary() {
            reporter.on_summary(&summary);
        }
        Ok(())
    }
}

/// What every round of one run has in common, so retiring one is a call rather
/// than a block repeated at each of the two places a round can finish.
struct Rounds<'a> {
    cfg: &'a ProfanityConfig,
    job: &'a Job,
    device_index: usize,
    seed: &'a Ulong4,
}

impl Rounds<'_> {
    /// Turn a completed round's buffers into hits. The caller has waited on the
    /// command buffer that wrote them.
    fn retire(
        &self,
        results: &ProtocolObject<dyn MTLBuffer>,
        flags: &ProtocolObject<dyn MTLBuffer>,
        passes: u64,
        best: &mut u64,
        reporter: &mut dyn Reporter,
    ) {
        let context = RoundContext {
            cfg: self.cfg,
            job: self.job,
            device_index: self.device_index,
            seed: self.seed,
            // `profanity_init` leaves every point one generator step ahead of
            // the scalar it was seeded with, and an iterate pass advances it
            // again before hashing, so after `passes` passes the address on
            // show belongs to `seed + passes + 1`. The OpenCL loop reaches the
            // same number a different way: it reads each pass's results at the
            // top of the next iteration, so its round counter lags one behind
            // its dispatches. Getting this wrong costs nothing visible — the
            // address is real and the offset well formed, it just names a key
            // to a different one.
            round: passes + 1,
        };

        // SAFETY: the buffer was allocated with SLOTS entries of this type and
        // the command buffer that wrote it has completed, which the caller has
        // waited on.
        let slots: Vec<ResultSlot> = unsafe { read_slots(results, SLOTS) };
        let found = if self.job.is_exact() {
            // SAFETY: the flag buffer holds SLOTS u32s and slot 0 is the
            // counter; the command buffer that wrote it has completed, so
            // nothing is still writing it.
            let total = unsafe { read_counter(flags) };
            context.drain_exact(&slots, total)
        } else {
            match context.take_best(&slots, *best) {
                Some((score, hit)) => {
                    *best = u64::from(score);
                    vec![Progress::Hit(hit)]
                }
                None => Vec::new(),
            }
        };

        for progress in found {
            progress.report(reporter);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_params_layout_matches_the_kernel() {
        // Three sets of four lanes, then six 32-bit fields.
        assert_eq!(size_of::<MtProfParams>(), 96 + 24);
    }
}
