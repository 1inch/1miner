//! create2, create3 and 1nft on Metal.
//!
//! Unlike the OpenCL path, the 200-byte pre-image arrives in a buffer rather
//! than as a compile-time constant, so changing deployer, code hash or base
//! salt does not force a pipeline rebuild.

use std::time::Instant;

use miner_core::SaltConfig;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{
    MetalBackend, clear, dispatch, encode, read_counter, read_slots, set_bytes, set_slice,
    threadgroup_width,
};
use crate::salt::{SaltRound, SaltSlot};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{EXACT_CAPACITY, Job, Progress, RESULT_SLOTS, Reporter, Result, kernels, wire};

const SLOTS: usize = RESULT_SLOTS;

#[repr(C)]
#[derive(Clone, Copy)]
struct MtParams {
    state: [u64; 25],
    device_index: u32,
    round: u32,
    second_hash: u32,
    score_max: u32,
    pattern_count: u32,
    exact_capacity: u32,
}

impl MetalBackend {
    pub(super) fn run_salt(
        &self,
        cfg: &SaltConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let queue = self.command_queue()?;

        let library = self.build_library(kernels::METAL_SALT)?;
        // --exact asks a different question and so runs a different kernel over
        // a different result layout; see the comment on `salt_iterate_exact`.
        let kernel_name = if job.is_exact() {
            "salt_iterate_exact"
        } else {
            "salt_iterate"
        };
        let pipeline = self.pipeline(&library, kernel_name)?;

        let results = self.shared_buffer(SLOTS * size_of::<SaltSlot>(), "result")?;
        let flags = self.shared_buffer(SLOTS * 4, "flag")?;

        let mode = wire::Mode::from(&job.score);
        let patterns = wire::patterns(job.exact.as_deref());

        let threadgroup = threadgroup_width(&pipeline, job.tuning.work_size);
        let reader = SaltRound {
            cfg,
            job,
            device_index: self.infos[0].index,
        };

        let start = Instant::now();
        let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);
        let mut round: u32 = 0;
        let mut best: u32 = 0;
        let mut hashes: u64 = 0;

        loop {
            if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
                break;
            }
            round = round.wrapping_add(1);

            // The exact kernel appends from the counter in foundFlags[0], so
            // clearing both buffers each round is the protocol rather than a
            // workaround: it is what lets the next round start at slot 1. The
            // previous round was waited on at the end of the loop body, so no
            // GPU work is in flight.
            if job.is_exact() {
                clear(&results);
                clear(&flags);
            }

            let params = MtParams {
                state: cfg.state_words(),
                device_index: 0,
                round,
                second_hash: cfg.mode.needs_second_hash().into(),
                score_max: best,
                pattern_count: patterns.len() as u32,
                exact_capacity: EXACT_CAPACITY as u32,
            };

            let (command_buffer, encoder) = encode(&queue)?;
            encoder.setComputePipelineState(&pipeline);
            // SAFETY: the indices match `salt_iterate`'s parameter positions in
            // kernels/metal/salt.metal, and both buffers plus `mode` and
            // `params` outlive the command buffer, which is waited on before
            // this iteration ends.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&results), 0, 0);
                // Buffer 1 is the scoring mode or the exact masks, chosen
                // alongside the kernel name above.
                if job.is_exact() {
                    set_slice(&encoder, &patterns, 1);
                } else {
                    set_bytes(&encoder, &mode, 1);
                }
                set_bytes(&encoder, &params, 2);
                encoder.setBuffer_offset_atIndex(Some(&flags), 0, 3);
            }
            dispatch(&encoder, job.tuning.round_size, threadgroup);
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            hashes += job.tuning.round_size as u64;

            // SAFETY: the buffer was allocated with SLOTS entries of this type
            // and the command buffer above has completed, so nothing is still
            // writing them.
            let slots: Vec<SaltSlot> = unsafe { read_slots(&results, SLOTS) };
            let found = if job.is_exact() {
                // SAFETY: the flag buffer holds SLOTS u32s and slot 0 is the
                // counter; as above, nothing is still writing it.
                let total = unsafe { read_counter(&flags) };
                reader.drain_exact(&slots, total)
            } else {
                match reader.take_best(&slots, u64::from(best)) {
                    Some((score, hit)) => {
                        best = score;
                        vec![Progress::Hit(hit)]
                    }
                    None => Vec::new(),
                }
            };
            for progress in found {
                progress.report(reporter);
            }

            meter.sample(hashes);
            let rate = meter.rate();
            reporter.on_speed(rate, &[rate]);
        }

        if let Some(summary) = meter.summary() {
            reporter.on_summary(&summary);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_params_layout_matches_the_kernel() {
        // 25 lanes plus six 32-bit fields.
        assert_eq!(size_of::<MtParams>(), 200 + 24);
    }
}
