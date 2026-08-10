//! create2, create3 and 1nft on Metal.
//!
//! Unlike the OpenCL path, the 200-byte pre-image arrives in a buffer rather
//! than as a compile-time constant, so changing deployer, code hash or base
//! salt does not force a pipeline rebuild.

use std::time::Instant;

use miner_core::{MineMode, SaltConfig, ScoreSpec};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};

use super::{MetalBackend, set_bytes, set_slice, threadgroup_width};
use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{
    BackendError, EXACT_CAPACITY, Hit, Job, MAX_SCORE, Progress, RESULT_SLOTS, Reporter, Result,
    kernels,
};

const SLOTS: usize = RESULT_SLOTS;

#[repr(C)]
#[derive(Clone, Copy)]
struct MtMode {
    function: u32,
    data1: [u8; 20],
    data2: [u8; 20],
}

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

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MtResult {
    salt: [u8; 32],
    hash: [u8; 20],
    found: u32,
}

/// One `--exact` mask, matching `Pattern` in kernels/metal/scoring.metal.
#[repr(C)]
#[derive(Clone, Copy)]
struct MtPattern {
    mask: [u8; 20],
    want: [u8; 20],
}

impl From<&ScoreSpec> for MtPattern {
    fn from(spec: &ScoreSpec) -> Self {
        Self {
            mask: spec.data1,
            want: spec.data2,
        }
    }
}

impl MetalBackend {
    pub(super) fn run_salt(
        &self,
        cfg: &SaltConfig,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()> {
        let device = &self.device;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| BackendError::Other("failed to create a Metal command queue".into()))?;

        let library = self.build_library(kernels::METAL_SALT)?;
        // --exact asks a different question and so runs a different kernel over
        // a different result layout; see the comment on `salt_iterate_exact`.
        let kernel_name = if job.is_exact() {
            "salt_iterate_exact"
        } else {
            "salt_iterate"
        };
        let pipeline = self.pipeline(&library, kernel_name)?;

        let results = device
            .newBufferWithLength_options(
                SLOTS * size_of::<MtResult>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| BackendError::Other("failed to allocate result buffer".into()))?;
        let flags = device
            .newBufferWithLength_options(SLOTS * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| BackendError::Other("failed to allocate flag buffer".into()))?;

        let mode = MtMode {
            function: job.score.function as u32,
            data1: job.score.data1,
            data2: job.score.data2,
        };
        let patterns: Vec<MtPattern> = job
            .exact
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(MtPattern::from)
            .collect();

        let threadgroup = threadgroup_width(&pipeline, job.tuning.work_size);

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
            // workaround: it is what lets the next round start at slot 1.
            if job.is_exact() {
                // SAFETY: both pointers come from `contents()` on a
                // StorageModeShared buffer, and the lengths are exactly the ones
                // the buffers were allocated with above. The previous round was
                // waited on at the end of the loop body, so no GPU work is in
                // flight and nothing else holds a reference to the storage.
                unsafe {
                    std::ptr::write_bytes(
                        results.contents().as_ptr().cast::<u8>(),
                        0,
                        SLOTS * size_of::<MtResult>(),
                    );
                    std::ptr::write_bytes(flags.contents().as_ptr().cast::<u8>(), 0, SLOTS * 4);
                }
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

            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| BackendError::Other("failed to create a command buffer".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| BackendError::Other("failed to create a compute encoder".into()))?;

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
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: job.tuning.round_size,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            hashes += job.tuning.round_size as u64;

            if job.is_exact() {
                // SAFETY: the flag buffer holds SLOTS u32s and slot 0 is the
                // counter; the command buffer above has completed, so nothing
                // is still writing either buffer.
                let total = unsafe { std::ptr::read_unaligned(flags.contents().as_ptr().cast()) };
                for found in read_exact(&results, cfg, job, total) {
                    match found {
                        Progress::Hit(hit) => reporter.on_hit(&hit),
                        Progress::Dropped {
                            count,
                            device_index,
                        } => reporter.on_dropped(count, device_index),
                    }
                }
            } else if let Some(hit) = read_best(&results, cfg, job, best) {
                best = hit.score;
                reporter.on_hit(&hit);
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

/// Build the hit a filled result slot describes, re-deriving the address from
/// the salt so a kernel that reconstructed the wrong one is caught.
fn hit_from(
    entry: &MtResult,
    score: u32,
    pattern: Option<usize>,
    cfg: &SaltConfig,
    job: &Job,
) -> Hit {
    let salt = entry.salt;
    let address = entry.hash;
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
        device_index: 0,
        verified: !job.verify || cfg.address_for_salt(&salt) == address,
    }
}

fn read_best(
    results: &ProtocolObject<dyn MTLBuffer>,
    cfg: &SaltConfig,
    job: &Job,
    best: u32,
) -> Option<Hit> {
    let base = results.contents().as_ptr() as *const MtResult;
    for score in (1..=MAX_SCORE).rev() {
        if score as u32 <= best {
            return None;
        }
        // SAFETY: the buffer was allocated with SLOTS entries of this type and
        // the GPU work that writes it has completed.
        let entry = unsafe { std::ptr::read_unaligned(base.add(score)) };
        if entry.found == 0 {
            continue;
        }
        return Some(hit_from(&entry, score as u32, None, cfg, job));
    }
    None
}

/// Slots are the round's matches in arrival order; `total` is how many there
/// were, including any the buffer had no room for.
fn read_exact(
    results: &ProtocolObject<dyn MTLBuffer>,
    cfg: &SaltConfig,
    job: &Job,
    total: u32,
) -> Vec<Progress> {
    let base = results.contents().as_ptr() as *const MtResult;
    let stored = (total as usize).min(EXACT_CAPACITY);
    let masks = job.exact.as_deref().unwrap_or_default();

    let mut found: Vec<Progress> = (1..=stored)
        .map(|slot| {
            // SAFETY: the buffer holds SLOTS entries and `stored` is capped at
            // EXACT_CAPACITY, which is below it; the GPU work has completed.
            let entry = unsafe { std::ptr::read_unaligned(base.add(slot)) };
            // `found` names the mask that matched in this layout, one-based so
            // an untouched slot is distinguishable from mask 0.
            let pattern = entry.found.saturating_sub(1) as usize;
            let score = masks.get(pattern).map_or(0, ScoreSpec::constrained_bytes);
            Progress::Hit(hit_from(&entry, score, Some(pattern), cfg, job))
        })
        .collect();

    if let Some(dropped) = total.checked_sub(stored as u32).filter(|d| *d > 0) {
        found.push(Progress::Dropped {
            count: dropped,
            device_index: 0,
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_kernel() {
        assert_eq!(size_of::<MtMode>(), 44);
        assert_eq!(size_of::<MtResult>(), 56);
        assert_eq!(size_of::<MtPattern>(), 40);
        // 25 lanes plus six 32-bit fields.
        assert_eq!(size_of::<MtParams>(), 200 + 24);
    }
}
