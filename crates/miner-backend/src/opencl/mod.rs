//! OpenCL backend: the default, and the only one that supports every mode.

pub mod profanity;
pub mod salt;

use std::path::PathBuf;

use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::program::Program;
use opencl3::types::cl_device_id;

use crate::{BackendError, DeviceInfo, Result};

pub(crate) fn cl_err<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> BackendError + '_ {
    move |e| BackendError::OpenCl(format!("{context}: {e:?}"))
}

/// A device handle that can cross a thread boundary.
///
/// `cl_device_id` is `*mut c_void`, so Rust treats it as neither `Send` nor
/// `Sync`. It is an opaque handle rather than a pointer we dereference, and the
/// OpenCL specification requires implementations to be thread safe, so passing
/// one to a worker thread is sound.
#[derive(Clone, Copy, Debug)]
pub struct DeviceId(pub cl_device_id);

unsafe impl Send for DeviceId {}
unsafe impl Sync for DeviceId {}

/// Every GPU the platform exposes, in platform order, with the requested
/// indices removed. Indices refer to the unfiltered list so that `--skip 1`
/// keeps meaning the same device after other devices are skipped.
pub fn enumerate_devices(skip: &[usize]) -> Result<Vec<(DeviceId, DeviceInfo)>> {
    let ids = get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(cl_err("failed to enumerate OpenCL GPU devices"))?;
    if ids.is_empty() {
        return Err(BackendError::NoDevices("opencl"));
    }
    if let Some(bad) = skip.iter().find(|i| **i >= ids.len()) {
        return Err(BackendError::BadDeviceIndex(*bad, ids.len()));
    }

    let mut out = Vec::new();
    for (index, id) in ids.into_iter().enumerate() {
        if skip.contains(&index) {
            continue;
        }
        let device = Device::new(id);
        out.push((
            DeviceId(id),
            DeviceInfo {
                index,
                name: device
                    .name()
                    .unwrap_or_else(|_| format!("OpenCL device {index}")),
                compute_units: device.max_compute_units().unwrap_or(0),
                global_memory: device.global_mem_size().unwrap_or(0),
            },
        ));
    }

    if out.is_empty() {
        return Err(BackendError::NoDevices("opencl"));
    }
    Ok(out)
}

fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("1miner").join("opencl"))
}

/// Cache key over everything that changes the compiled output. The kernel
/// constants are part of `options`, so a new deployer or salt yields a new key
/// and can never reuse a stale binary.
fn cache_key(source: &str, options: &str, device_name: &str) -> String {
    let digest = miner_core::keccak256(
        format!("{device_name}\u{0}{options}\u{0}{source}").as_bytes(),
    );
    hex::encode(&digest[..16])
}

/// Build a program, reusing a cached device binary when one matches.
///
/// Falling back to a source build on any cache problem is deliberate: a stale
/// or corrupt cache should cost a recompile, never a wrong kernel.
pub fn build_program(
    context: &Context,
    device: &Device,
    source: &str,
    options: &str,
    use_cache: bool,
) -> Result<Program> {
    let device_name = device.name().unwrap_or_default();
    let cached_path = if use_cache {
        cache_dir().map(|d| d.join(format!("{}.bin", cache_key(source, options, &device_name))))
    } else {
        None
    };

    if let Some(path) = &cached_path
        && let Ok(binary) = std::fs::read(path)
        && let Ok(program) =
            Program::create_and_build_from_binary(context, &[binary.as_slice()], options)
    {
        return Ok(program);
    }

    let program = Program::create_and_build_from_source(context, source, options)
        .map_err(BackendError::Build)?;

    if let Some(path) = &cached_path
        && let Ok(binaries) = program.get_binaries()
        && let Some(binary) = binaries.first()
        && let Some(parent) = path.parent()
        && std::fs::create_dir_all(parent).is_ok()
    {
        let _ = std::fs::write(path, binary);
    }

    Ok(program)
}

/// Format the 200-byte keccak state as the comma-separated ulong initialiser
/// the kernels expect as a `-D` define.
pub fn state_define(words: &[u64; 25]) -> String {
    words
        .iter()
        .map(|w| format!("0x{w:x}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_tracks_every_input() {
        let a = cache_key("src", "-D A=1", "GPU");
        assert_eq!(a, cache_key("src", "-D A=1", "GPU"));
        assert_ne!(a, cache_key("src2", "-D A=1", "GPU"));
        assert_ne!(a, cache_key("src", "-D A=2", "GPU"));
        assert_ne!(a, cache_key("src", "-D A=1", "other GPU"));
    }

    #[test]
    fn state_define_is_little_endian_hex() {
        let mut words = [0u64; 25];
        words[0] = 0xdead_beef;
        words[24] = 1;
        let s = state_define(&words);
        assert!(s.starts_with("0xdeadbeef,"));
        assert!(s.ends_with(",0x1"));
        assert_eq!(s.split(',').count(), 25);
    }
}
