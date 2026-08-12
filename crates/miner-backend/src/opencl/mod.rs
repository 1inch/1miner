//! OpenCL backend: the default, and the only one that supports every mode.

pub mod profanity;
pub mod salt;

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::memory::Buffer;
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_mem_flags};

use crate::speed::{DEFAULT_WINDOW, SpeedMeter, combine};
use crate::{BackendError, DeviceInfo, Job, Progress, Reporter, Result, drain_hits};

/// How often the dispatcher looks at what the device threads have produced.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

pub(crate) fn cl_err<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> BackendError + '_ {
    move |e| BackendError::OpenCl(format!("{context}: {e:?}"))
}

/// A device handle that can cross a thread boundary.
///
/// `cl_device_id` is `*mut c_void`, so Rust treats it as neither `Send` nor
/// `Sync`.
#[derive(Clone, Copy, Debug)]
pub struct DeviceId(pub cl_device_id);

// SAFETY: the handle is opaque — nothing in this crate dereferences it, it is
// only handed back to the driver — and the OpenCL specification requires
// implementations to be thread safe, so moving one to a worker thread is sound.
unsafe impl Send for DeviceId {}
// SAFETY: as for `Send` above. Sharing the handle is what lets each device
// thread hold its own context while the dispatcher keeps the list.
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
                driver: device.driver_version().ok(),
            },
        ));
    }

    if out.is_empty() {
        return Err(BackendError::NoDevices("opencl"));
    }
    Ok(out)
}

/// Allocate a buffer of `len` elements that OpenCL owns.
///
/// `what` names it in the error, since a device that cannot fit an allocation
/// should say which one, and the scratch buffers a profanity round works in are
/// the ones large enough to fail.
pub(crate) fn buffer<T>(
    context: &Context,
    flags: cl_mem_flags,
    len: usize,
    what: &str,
) -> Result<Buffer<T>> {
    // SAFETY: a null host pointer with neither CL_MEM_USE_HOST_PTR nor
    // CL_MEM_COPY_HOST_PTR set asks OpenCL to own the allocation, so there is no
    // host memory whose lifetime has to be upheld here.
    unsafe { Buffer::<T>::create(context, flags, len, std::ptr::null_mut()) }
        .map_err(|e| BackendError::OpenCl(format!("failed to allocate the {what} buffer: {e:?}")))
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
    let digest =
        miner_core::keccak256(format!("{device_name}\u{0}{options}\u{0}{source}").as_bytes());
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

/// What one device thread reports into.
///
/// A device cannot reach the reporter itself — several run at once and the
/// reporter is not shared — so findings queue up here and the dispatcher's poll
/// loop drains them.
pub struct DeviceRun<'a> {
    /// Hashes this device has tried, read by the speed meter.
    pub counter: &'a AtomicU64,
    pub hits: &'a Mutex<Vec<Progress>>,
    /// The best score any device has reached, shared so a strong hit on one GPU
    /// raises the bar on all of them. The exact path has no bar and leaves this
    /// at zero.
    pub best_score: &'a AtomicU64,
}

/// One thread per GPU, each with its own context and queue, polled for progress
/// until the job's duration elapses or `should_stop` returns true.
///
/// Both modes enumerate candidates differently but are dispatched identically,
/// so `device` is the only part either one writes: it runs one GPU until the
/// job ends, and whichever error surfaces first is the one the run reports.
fn run_devices(
    ids: &[DeviceId],
    infos: &[DeviceInfo],
    job: &Job,
    reporter: &mut dyn Reporter,
    should_stop: &(dyn Fn() -> bool + Sync),
    device: impl Fn(DeviceId, &DeviceInfo, DeviceRun<'_>, Instant) -> Result<()> + Sync,
) -> Result<()> {
    let counters: Vec<AtomicU64> = ids.iter().map(|_| AtomicU64::new(0)).collect();
    let hits: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
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
            let (counters, hits, best_score) = (&counters, &hits, &best_score);
            let (failure, device) = (&failure, &device);

            scope.spawn(move || {
                let run = DeviceRun {
                    counter: &counters[slot],
                    hits,
                    best_score,
                };
                if let Err(e) = device(*device_id, info, run, start) {
                    let mut guard = failure.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(e);
                    }
                }
            });
        }

        // Poll for progress while the device threads work.
        loop {
            std::thread::sleep(POLL_INTERVAL);
            reported = drain_hits(&hits, reported, reporter);

            for (meter, counter) in meters.iter_mut().zip(counters.iter()) {
                meter.sample(counter.load(Ordering::Relaxed));
            }
            let per_device: Vec<f64> = meters.iter().map(SpeedMeter::rate).collect();
            reporter.on_speed(per_device.iter().sum(), &per_device);

            if job.expired(start) || should_stop() || failure.lock().unwrap().is_some() {
                break;
            }
        }
    });

    // Anything found between the last poll and shutdown.
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
