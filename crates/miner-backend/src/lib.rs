//! Compute backends for 1miner.
//!
//! A backend runs one [`Job`] and reports [`Hit`]s. The address maths lives in
//! `miner-core`; a backend only decides how candidates are enumerated and on
//! what hardware.

pub mod cpu;
pub mod kernels;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(feature = "opencl")]
pub mod opencl;

use std::time::Duration;

use miner_core::{Address, ModeConfig, Salt, ScoreSpec};

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("no compute devices found for backend {0}")]
    NoDevices(&'static str),
    #[error("device index {0} is out of range ({1} devices present)")]
    BadDeviceIndex(usize, usize),
    #[error("{0} backend does not support {1} mode")]
    Unsupported(&'static str, &'static str),
    #[error("kernel build failed: {0}")]
    Build(String),
    #[error("opencl error: {0}")]
    OpenCl(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// Which Keccak implementation to compile in. Both are functionally identical,
/// so they can be raced against each other on any given GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeccakVariant {
    /// ERADICATE2/3's tuned permutation: fused theta, rotated chi step.
    #[default]
    Tuned,
    /// profanity2's more literal permutation.
    Plain,
}

impl KeccakVariant {
    pub fn source(self) -> &'static str {
        match self {
            KeccakVariant::Tuned => kernels::KECCAK_TUNED,
            KeccakVariant::Plain => kernels::KECCAK_PLAIN,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KeccakVariant::Tuned => "tuned",
            KeccakVariant::Plain => "plain",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "tuned" => Some(KeccakVariant::Tuned),
            "plain" => Some(KeccakVariant::Plain),
            _ => None,
        }
    }

    pub fn all() -> &'static [KeccakVariant] {
        &[KeccakVariant::Tuned, KeccakVariant::Plain]
    }
}

/// Tuning knobs. Defaults match the reference miners so published hashrates
/// stay comparable.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// OpenCL local work size. 0 lets the implementation choose.
    pub work_size: usize,
    /// Largest single enqueue; a round is split into chunks of this size.
    pub work_max: Option<usize>,
    /// Salt candidates attempted per round, per device.
    pub round_size: usize,
    /// profanity batched-inverse width.
    pub inverse_size: usize,
    /// profanity number of parallel inverse batches.
    pub inverse_multiple: usize,
    pub skip_devices: Vec<usize>,
    pub no_cache: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            work_size: 128,
            work_max: None,
            round_size: 16_777_216,
            inverse_size: 255,
            inverse_multiple: 16_384,
            skip_devices: Vec::new(),
            no_cache: false,
        }
    }
}

impl Tuning {
    /// Work-items per profanity round.
    pub fn profanity_round_size(&self) -> usize {
        self.inverse_size * self.inverse_multiple
    }
}

/// A configured search.
#[derive(Debug, Clone)]
pub struct Job {
    pub mode: ModeConfig,
    pub score: ScoreSpec,
    pub keccak: KeccakVariant,
    pub tuning: Tuning,
    /// Stop after this long. `None` runs until interrupted.
    pub duration: Option<Duration>,
    /// Re-derive every hit on the CPU before reporting it.
    pub verify: bool,
    /// All-or-nothing mode (`--exact`). When set, only candidates reaching this
    /// score are reported, the bar never rises, and every subsequent match is
    /// reported rather than just improvements on the best seen.
    pub exact_score: Option<u32>,
}

impl Job {
    /// The score a candidate must beat to be worth recording, and whether the
    /// bar rises as better candidates arrive.
    ///
    /// Ordinary scoring keeps the best found so far, so the bar climbs and each
    /// line printed is an improvement. `--exact` pins the bar just below the
    /// full match instead, so the kernel records nothing else and every full
    /// match gets reported.
    pub fn initial_threshold(&self) -> u32 {
        self.exact_score.map_or(0, |n| n.saturating_sub(1))
    }

    pub fn is_exact(&self) -> bool {
        self.exact_score.is_some()
    }
}

/// A candidate that beat the previous best score.
#[derive(Debug, Clone)]
pub struct Hit {
    pub score: u32,
    pub address: Address,
    /// Salt modes: the full 32-byte salt.
    pub salt: Option<Salt>,
    /// 1nft: the mined 16-byte magic, the high half of the salt.
    pub magic: Option<[u8; 16]>,
    /// profanity: the offset to add to the seed private key.
    pub offset: Option<[u8; 32]>,
    pub device_index: usize,
    /// Whether the CPU agreed this address follows from the reported inputs.
    pub verified: bool,
}

/// Progress callbacks. Both are invoked from the run loop.
pub trait Reporter: Send {
    fn on_hit(&mut self, hit: &Hit);
    /// Aggregate hashrate in hashes per second.
    fn on_speed(&mut self, total: f64, per_device: &[f64]);
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub compute_units: u32,
    pub global_memory: u64,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn devices(&self) -> &[DeviceInfo];
    /// Run until the job's duration elapses or `should_stop` returns true.
    fn run(
        &mut self,
        job: &Job,
        reporter: &mut dyn Reporter,
        should_stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<()>;
}
