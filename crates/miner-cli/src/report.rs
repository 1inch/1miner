//! Terminal output: a live speed line plus a printed line per improved hit.

use std::io::Write;
use std::time::Instant;

use miner_backend::speed::SpeedSummary;
use miner_backend::{Hit, Reporter};
use miner_core::{MineMode, to_checksum_address};

const CLEAR_LINE: &str = "\x1b[2K\r";

pub struct TerminalReporter {
    mode: MineMode,
    start: Instant,
    quiet: bool,
    pub hits: usize,
    pub unverified: usize,
}

impl TerminalReporter {
    pub fn new(mode: MineMode, quiet: bool) -> Self {
        Self {
            mode,
            start: Instant::now(),
            quiet,
            hits: 0,
            unverified: 0,
        }
    }
}

impl Reporter for TerminalReporter {
    fn on_hit(&mut self, hit: &Hit) {
        self.hits += 1;
        let seconds = self.start.elapsed().as_secs();

        let detail = match self.mode {
            MineMode::Nft => hit
                .magic
                .map(|m| format!("Magic: 0x{}", hex::encode(m)))
                .unwrap_or_default(),
            MineMode::Profanity => hit
                .offset
                .map(|o| format!("Offset: 0x{}", hex::encode(o)))
                .unwrap_or_default(),
            _ => hit
                .salt
                .map(|s| format!("Salt: 0x{}", hex::encode(s)))
                .unwrap_or_default(),
        };

        // A hit that fails re-derivation means the kernel and the CPU disagree,
        // which is a correctness bug rather than a lucky find. Say so loudly
        // instead of printing it like any other result.
        let flag = if hit.verified {
            String::new()
        } else {
            self.unverified += 1;
            "  [UNVERIFIED: CPU re-derivation disagrees with the kernel]".to_string()
        };

        print!("{CLEAR_LINE}");
        println!(
            "  Time: {seconds:>5}s  Score: {:>2}  GPU{}  {detail}  Address: {}{flag}",
            hit.score,
            hit.device_index,
            to_checksum_address(&hit.address),
        );
        let _ = std::io::stdout().flush();
    }

    /// The figure to quote. `on_speed` is a short rolling window, so it moves
    /// around; this is the average over everything after the warmup.
    fn on_summary(&mut self, summary: &SpeedSummary) {
        print!("{CLEAR_LINE}");
        println!(
            "Measured: {:.3} MH/s over {:.1}s ({} hashes)",
            summary.rate / 1.0e6,
            summary.elapsed.as_secs_f64(),
            summary.hashes,
        );
        let _ = std::io::stdout().flush();
    }

    fn on_speed(&mut self, total: f64, per_device: &[f64]) {
        if self.quiet {
            return;
        }
        let mut line = format!("{CLEAR_LINE}Speed: {:.3} MH/s", total / 1.0e6);
        if per_device.len() > 1 {
            for (i, speed) in per_device.iter().enumerate() {
                line.push_str(&format!("  GPU{i}: {:.3} MH/s", speed / 1.0e6));
            }
        }
        print!("{line}");
        let _ = std::io::stdout().flush();
    }
}
