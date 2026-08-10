//! Metal backend for macOS.
//!
//! Apple deprecated OpenCL, and on Apple silicon it is both slower and capped
//! at OpenCL 1.2, so Metal is the fast local path. All four modes are covered:
//! the keccak-based salt modes in [`salt`] and the secp256k1 search in
//! [`profanity`].
//!
//! There is one system default device, so this runs single threaded.
//!
//! A Metal library is compiled from one source string at run time, so each
//! kernel's source is the shared prelude concatenated with its own. There is no
//! equivalent of OpenCL's `-D`, and no on-disk cache of the compiled result;
//! everything a job varies arrives in a buffer instead.

mod profanity;
mod salt;

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary,
};

use crate::{Backend, BackendError, DeviceInfo, Job, ModeConfig, Reporter, Result, kernels};

/// The prelude every Metal library starts with. `METAL_KECCAK` carries the
/// `#include` and so has to come first.
fn library_source(kernel: &str) -> String {
    format!(
        "{}\n{}\n{}",
        kernels::METAL_KECCAK,
        kernels::METAL_SCORING,
        kernel
    )
}

pub struct MetalBackend {
    infos: Vec<DeviceInfo>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(BackendError::NoDevices("metal"))?;
        let infos = vec![DeviceInfo {
            index: 0,
            name: device.name().to_string(),
            compute_units: 0,
            global_memory: device.recommendedMaxWorkingSetSize(),
        }];
        Ok(Self { infos, device })
    }

    /// Compile a kernel with the shared prelude in front of it.
    fn build_library(&self, kernel: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let source = NSString::from_str(&library_source(kernel));
        self.device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| BackendError::Build(format!("Metal kernel failed to compile: {e:?}")))
    }

    fn pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| BackendError::Build(format!("{name} not found in library")))?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| BackendError::Build(format!("pipeline creation failed for {name}: {e:?}")))
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
        match &job.mode {
            ModeConfig::Salt(cfg) => self.run_salt(cfg, job, reporter, should_stop),
            ModeConfig::Profanity(cfg) => self.run_profanity(cfg, job, reporter, should_stop),
        }
    }
}

/// The threadgroup width to dispatch with, bounded by what the pipeline can
/// take. `--work 0` leaves the choice here rather than to the driver, since
/// Metal has no equivalent of OpenCL's null local size.
fn threadgroup_width(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    requested: usize,
) -> usize {
    pipeline
        .maxTotalThreadsPerThreadgroup()
        .min(if requested > 0 { requested } else { 256 })
        .max(1)
}

/// Push a small struct straight into the command encoder rather than
/// allocating a buffer for it.
///
/// # Safety
///
/// `T` must have the layout the kernel expects at `index`, which means a
/// `#[repr(C)]` type matching the corresponding parameter in the kernel source.
unsafe fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    let ptr =
        NonNull::new(std::ptr::from_ref(value) as *mut c_void).expect("reference is never null");
    // SAFETY: `ptr` points at `value`, which outlives the call, and the length
    // is exactly its size. Metal copies the bytes into the command buffer, so
    // the borrow does not have to outlive the encoding.
    unsafe { encoder.setBytes_length_atIndex(ptr, size_of::<T>(), index) };
}

/// As `set_bytes`, for an array the kernel indexes.
///
/// # Safety
///
/// `T` must have the layout the kernel expects at `index`, and the slice must
/// hold at least as many elements as the kernel will read.
unsafe fn set_slice<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    values: &[T],
    index: usize,
) {
    let ptr = NonNull::new(values.as_ptr() as *mut c_void).expect("slice pointer is never null");
    // SAFETY: as `set_bytes`, with the length covering the whole slice, which
    // outlives the call because Metal copies it into the command buffer.
    unsafe { encoder.setBytes_length_atIndex(ptr, size_of_val(values), index) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prelude has to arrive before the kernel that calls into it, and the
    /// whole library has to be one string, so the order here is the contract.
    #[test]
    fn the_library_source_carries_the_prelude_before_the_kernel() {
        for (kernel, entry_point) in [
            (kernels::METAL_SALT, "kernel void salt_iterate"),
            (kernels::METAL_PROFANITY, "kernel void profanity_init"),
        ] {
            let source = library_source(kernel);
            let include = source.find("#include <metal_stdlib>").expect("no prelude");
            let keccak = source.find("static void keccakf").expect("no keccak");
            let scoring = source.find("int score_address").expect("no scorer");
            let entry = source.find(entry_point).expect("no entry point");
            assert!(include < keccak && keccak < scoring && scoring < entry);
        }
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
            assert!(
                kernels::METAL_SCORING.contains(&expected),
                "missing or wrong: {expected}"
            );
        }
    }
}
