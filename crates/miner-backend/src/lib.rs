//! Compute backends for 1miner.
//!
//! A backend runs one [`Job`] and reports [`Hit`]s. The address maths lives in
//! `miner-core`; a backend only decides how candidates are enumerated and on
//! what hardware.

pub mod cpu;
pub mod kernels;
pub mod profanity;
pub mod salt;
pub mod speed;
pub mod wire;

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
    /// Excluded from the measured summary, so a benchmark figure does not
    /// include kernel compilation and the first slow round.
    pub warmup: Duration,
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
            warmup: Duration::ZERO,
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
    /// The masks `--exact` searches for, or `None` for ordinary scoring.
    ///
    /// This selects a different question and therefore a different kernel and a
    /// different result layout, rather than a variation on scoring: scoring
    /// asks for the best candidate so far, and `--exact` asks for everyone who
    /// matched. Only `data1` and `data2` of each spec are read; the function
    /// is not, since the exact kernels do not score.
    pub exact: Option<Vec<ScoreSpec>>,
}

impl Job {
    pub fn is_exact(&self) -> bool {
        self.exact.is_some()
    }
}

/// How many matches one round on one device can hand back in `--exact`.
///
/// Everything recovered is reported, and a round that finds more says by how
/// many it overflowed, so this one number bounds both the buffer and the output.
/// It is deliberately not `MAX_SCORE`, which is 40 because an address has 40
/// nibbles and has nothing to say about how many matches a round can hold.
pub const EXACT_CAPACITY: usize = 256;

/// Slots in a result buffer: one counter plus whichever layout is in use.
pub const RESULT_SLOTS: usize = 1 + if MAX_SCORE > EXACT_CAPACITY {
    MAX_SCORE
} else {
    EXACT_CAPACITY
};

/// Highest score a result buffer has a slot for, which is every nibble of an
/// address matching.
pub const MAX_SCORE: usize = 40;

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
    /// `--exact`: which of the masks this address matched. `score` is the same
    /// number for every match there and so says nothing; this is what does.
    pub pattern: Option<usize>,
    pub device_index: usize,
    /// Whether the CPU agreed this address follows from the reported inputs.
    pub verified: bool,
}

/// What a device thread hands back to the run loop.
///
/// A device cannot reach the reporter directly — several of them run at once
/// and the reporter is not shared — so findings queue up and the run loop
/// drains them. Drops travel in the same queue as hits rather than in a counter
/// of their own, which is what keeps a round's overflow line beside the round
/// that overflowed.
#[derive(Debug, Clone)]
pub enum Progress {
    Hit(Hit),
    /// Matches this round found beyond what the result buffer could keep.
    Dropped {
        count: u32,
        device_index: usize,
    },
}

impl Progress {
    /// Hand one finding to the reporter.
    ///
    /// Both queued and immediate reporting go through here so that a backend
    /// which starts dropping matches cannot be the one that forgets to say so:
    /// adding a variant to `Progress` without a callback for it stops being
    /// something four separate `match` arms could each miss.
    pub fn report(&self, reporter: &mut dyn Reporter) {
        match self {
            Progress::Hit(hit) => reporter.on_hit(hit),
            Progress::Dropped {
                count,
                device_index,
            } => reporter.on_dropped(*count, *device_index),
        }
    }
}

/// Progress callbacks, invoked from the run loop.
pub trait Reporter: Send {
    fn on_hit(&mut self, hit: &Hit);
    /// Aggregate hashrate over the recent window, in hashes per second.
    fn on_speed(&mut self, total: f64, per_device: &[f64]);
    /// The post-warmup average, reported once when the run ends. This is the
    /// figure a benchmark should quote, since the live rate is a short window.
    fn on_summary(&mut self, _summary: &crate::speed::SpeedSummary) {}
    /// Matches a round found beyond what the result buffer could keep. Reported
    /// so that a run which is discarding results says so rather than looking
    /// like one where matches are simply rare.
    fn on_dropped(&mut self, _count: u32, _device_index: usize) {}
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
