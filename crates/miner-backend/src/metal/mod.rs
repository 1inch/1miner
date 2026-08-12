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
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::{Backend, BackendError, DeviceInfo, Job, ModeConfig, Reporter, Result, kernels};

/// Rounds that can be in flight at once, and so result buffers to rotate
/// through. Two is enough to keep the GPU fed: the rounds are strictly ordered
/// on the device anyway, and all a third would buy is a longer wait before a
/// hit is printed.
const PIPELINE: usize = 2;

/// Slots in one result buffer, and so in the flag buffer beside it.
const SLOTS: usize = crate::RESULT_SLOTS;

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
            // Metal ships with the OS rather than with a driver of its own, so
            // the macOS version is what a Metal figure has to be recorded
            // against; there is nothing to ask the device for.
            driver: None,
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

    fn command_queue(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
        self.device
            .newCommandQueue()
            .ok_or_else(|| BackendError::Other("failed to create a Metal command queue".into()))
    }

    /// A buffer the CPU can read, which is what results and their counters have
    /// to be. The scratch buffers a search works in stay in private storage and
    /// are allocated where their size can be explained.
    fn shared_buffer(
        &self,
        bytes: usize,
        what: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| BackendError::Other(format!("failed to allocate the {what} buffer")))
    }
}

type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// One command buffer with one compute encoder, which is how every dispatch
/// here is submitted.
fn encode(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<(CommandBuffer, Encoder)> {
    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| BackendError::Other("failed to create a command buffer".into()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| BackendError::Other("failed to create a compute encoder".into()))?;
    Ok((command_buffer, encoder))
}

/// A one-dimensional dispatch, which is the only shape any kernel here wants.
fn dispatch(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, threads: usize, group: usize) {
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group,
            height: 1,
            depth: 1,
        },
    );
}

/// Zero a shared buffer in full.
///
/// The caller must have waited on any command buffer that uses it: this writes
/// storage the GPU can be reading, and clearing a buffer a round in flight is
/// still appending to would lose that round's matches.
fn clear(buffer: &ProtocolObject<dyn MTLBuffer>) {
    // SAFETY: the pointer comes from `contents()` on a StorageModeShared
    // buffer and the length is the one it was allocated with, so the write is
    // in bounds. No GPU work is in flight, which is the caller's obligation
    // above.
    unsafe {
        std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, buffer.length());
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

/// The `--exact` match count, which both exact kernels keep in slot 0 of their
/// flag buffer and which counts past what the result buffer could hold.
///
/// # Safety
///
/// The buffer must hold at least one `u32`, and the command buffer that wrote
/// it must have completed.
unsafe fn read_counter(buffer: &ProtocolObject<dyn MTLBuffer>) -> u32 {
    // SAFETY: the caller guarantees the buffer holds the counter and that
    // nothing is still writing it.
    unsafe { std::ptr::read_unaligned(buffer.contents().as_ptr().cast()) }
}

/// Copy a result buffer out as owned slots, so the round can be read without
/// holding a borrow of shared storage the next round will overwrite.
///
/// # Safety
///
/// The buffer must hold at least `count` elements of `T`, and the command
/// buffer that wrote them must have completed.
unsafe fn read_slots<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<T> {
    let base = buffer.contents().as_ptr().cast::<T>();
    (0..count)
        // SAFETY: the caller guarantees the buffer holds `count` elements and
        // that nothing is still writing them. The read is unaligned because a
        // slot layout the kernels fix does not promise `T`'s alignment.
        .map(|i| unsafe { std::ptr::read_unaligned(base.add(i)) })
        .collect()
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
