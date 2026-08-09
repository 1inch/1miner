//! Metal backend for the salt modes on macOS.
//!
//! Apple deprecated OpenCL, and on Apple silicon it is both slower and capped
//! at OpenCL 1.2, so Metal is the fast local path. Only the keccak-based salt
//! modes are supported; profanity needs a secp256k1 kernel that does not exist
//! for Metal yet, and OpenCL remains the backend that covers every mode.
//!
//! There is one system default device, so this runs single threaded.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

use miner_core::{MineMode, ModeConfig, SaltConfig};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::speed::{DEFAULT_WINDOW, SpeedMeter};
use crate::{Backend, BackendError, DeviceInfo, Hit, Job, Reporter, Result};

pub const MAX_SCORE: usize = 40;
const SLOTS: usize = MAX_SCORE + 1;

pub const SALT_SOURCE: &str = include_str!("../../../kernels/metal/salt.metal");

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
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MtResult {
    salt: [u8; 32],
    hash: [u8; 20],
    found: u32,
}

pub struct MetalBackend {
    infos: Vec<DeviceInfo>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or(BackendError::NoDevices("metal"))?;
        let infos = vec![DeviceInfo {
            index: 0,
            name: device.name().to_string(),
            compute_units: 0,
            global_memory: device.recommendedMaxWorkingSetSize(),
        }];
        Ok(Self { infos, device })
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
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
            return Err(BackendError::Unsupported("metal", "profanity"));
        };
        self.run_salt(cfg, job, reporter, should_stop)
    }
}

impl MetalBackend {
    fn run_salt(
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

        let source = NSString::from_str(SALT_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| BackendError::Build(format!("Metal kernel failed to compile: {e:?}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("salt_iterate"))
            .ok_or_else(|| BackendError::Build("salt_iterate not found in library".into()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| BackendError::Build(format!("pipeline creation failed: {e:?}")))?;

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

        let threadgroup = pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(if job.tuning.work_size > 0 { job.tuning.work_size } else { 256 })
            .max(1);

        let start = Instant::now();
        let mut meter = SpeedMeter::starting_at(start, DEFAULT_WINDOW, job.tuning.warmup);
        let mut round: u32 = 0;
        let mut best: u32 = job.initial_threshold();
        let mut hashes: u64 = 0;

        loop {
            if should_stop() || job.duration.is_some_and(|d| start.elapsed() >= d) {
                break;
            }
            round = round.wrapping_add(1);

            // The kernel keeps one slot per score and one flag per slot, so in
            // --exact mode both are cleared each round; otherwise a repeat
            // match at the same score would be silently dropped.
            if job.is_exact() {
                unsafe {
                    std::ptr::write_bytes(
                        results.contents().as_ptr() as *mut u8,
                        0,
                        SLOTS * size_of::<MtResult>(),
                    );
                    std::ptr::write_bytes(flags.contents().as_ptr() as *mut u8, 0, SLOTS * 4);
                }
            }

            let params = MtParams {
                state: cfg.state_words(),
                device_index: 0,
                round,
                second_hash: cfg.mode.needs_second_hash().into(),
                score_max: best,
            };

            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| BackendError::Other("failed to create a command buffer".into()))?;
            let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                BackendError::Other("failed to create a compute encoder".into())
            })?;

            encoder.setComputePipelineState(&pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&results), 0, 0);
                set_bytes(&encoder, &mode, 1);
                set_bytes(&encoder, &params, 2);
                encoder.setBuffer_offset_atIndex(Some(&flags), 0, 3);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: job.tuning.round_size, height: 1, depth: 1 },
                MTLSize { width: threadgroup, height: 1, depth: 1 },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            hashes += job.tuning.round_size as u64;

            if let Some(hit) = read_best(&results, cfg, job, best) {
                if !job.is_exact() {
                    best = hit.score;
                }
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

/// Push a small struct straight into the command encoder rather than
/// allocating a buffer for it.
unsafe fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    let ptr = NonNull::new(std::ptr::from_ref(value) as *mut c_void)
        .expect("reference is never null");
    unsafe { encoder.setBytes_length_atIndex(ptr, size_of::<T>(), index) };
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

        let salt = entry.salt;
        let address = entry.hash;
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
            device_index: 0,
            verified,
        });
    }
    None
}

/// Metal supports the salt modes only.
pub fn supports(mode: MineMode) -> bool {
    mode.is_salt_mode()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_kernel() {
        assert_eq!(size_of::<MtMode>(), 44);
        assert_eq!(size_of::<MtResult>(), 56);
        // 25 lanes plus four 32-bit fields.
        assert_eq!(size_of::<MtParams>(), 200 + 16);
    }

    #[test]
    fn only_salt_modes_are_supported() {
        assert!(supports(MineMode::Create2));
        assert!(supports(MineMode::Create3));
        assert!(supports(MineMode::Nft));
        assert!(!supports(MineMode::Profanity));
    }

    #[test]
    fn kernel_source_declares_the_entry_point() {
        assert!(SALT_SOURCE.contains("kernel void salt_iterate"));
    }

    /// The scoring constants are duplicated in the Metal source, so keep them
    /// pinned to the shared enum.
    #[test]
    fn scoring_constants_agree_with_the_enum() {
        use miner_core::ScoreFn;
        for (name, value) in [
            ("kBenchmark", ScoreFn::Benchmark as u32),
            ("kZeroBytes", ScoreFn::ZeroBytes as u32),
            ("kMatching", ScoreFn::Matching as u32),
            ("kLeading", ScoreFn::Leading as u32),
            ("kRange", ScoreFn::Range as u32),
            ("kMirror", ScoreFn::Mirror as u32),
            ("kDoubles", ScoreFn::Doubles as u32),
            ("kLeadingRange", ScoreFn::LeadingRange as u32),
        ] {
            let expected = format!("constant uint {name} = {value};");
            assert!(SALT_SOURCE.contains(&expected), "missing or wrong: {expected}");
        }
    }
}
