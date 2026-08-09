//! Hashrate measurement.
//!
//! Rates are reported over a rolling window rather than as an average since the
//! run began. The distinction matters more than it sounds: a cumulative average
//! takes a long time to forget a slow first round, so it creeps upwards for
//! tens of seconds and understates the real rate, and it can never show a GPU
//! that has started to throttle. On a rented machine that second property is
//! the one you actually want.
//!
//! Samples are cumulative counter readings taken by the polling loop, so a
//! backend only has to hand over the total it already tracks.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How much recent history the reported rate covers.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(3);

/// Rates below this span are meaningless, so they are reported as zero rather
/// than as a huge number derived from two nearly simultaneous readings.
const MIN_SPAN: Duration = Duration::from_millis(100);

/// The average over everything after the warmup, for benchmarks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedSummary {
    /// Hashes per second.
    pub rate: f64,
    pub hashes: u64,
    pub elapsed: Duration,
}

#[derive(Debug)]
pub struct SpeedMeter {
    /// `(when, cumulative total)`, trimmed to `window`.
    samples: VecDeque<(Instant, u64)>,
    window: Duration,
    /// Readings before this are excluded from the summary.
    measured_after: Instant,
    /// First reading at or after `measured_after`.
    measured_base: Option<(Instant, u64)>,
    latest: Option<(Instant, u64)>,
}

impl SpeedMeter {
    pub fn new(window: Duration, warmup: Duration) -> Self {
        Self::starting_at(Instant::now(), window, warmup)
    }

    pub fn starting_at(start: Instant, window: Duration, warmup: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window,
            measured_after: start + warmup,
            measured_base: None,
            latest: None,
        }
    }

    /// Record a cumulative counter reading.
    pub fn sample(&mut self, total: u64) {
        self.sample_at(Instant::now(), total);
    }

    pub fn sample_at(&mut self, now: Instant, total: u64) {
        self.samples.push_back((now, total));
        // Keep the newest reading that is at least a window old, so the span
        // covers the whole window, and drop everything older than that. Two
        // readings are always kept so a rate survives a long pause.
        while self.samples.len() > 2 && now.duration_since(self.samples[1].0) >= self.window {
            self.samples.pop_front();
        }

        if self.measured_base.is_none() && now >= self.measured_after {
            self.measured_base = Some((now, total));
        }
        self.latest = Some((now, total));
    }

    /// Hashes per second over the recent window.
    ///
    /// A device that has stopped producing work reads zero within one window,
    /// because the zero deltas simply fill it up. That needs no separate stall
    /// timeout.
    pub fn rate(&self) -> f64 {
        let (Some(&(first_at, first)), Some(&(last_at, last))) =
            (self.samples.front(), self.samples.back())
        else {
            return 0.0;
        };
        let span = last_at.duration_since(first_at);
        if span < MIN_SPAN {
            return 0.0;
        }
        last.saturating_sub(first) as f64 / span.as_secs_f64()
    }

    /// The post-warmup average, or `None` before the warmup has elapsed or
    /// while too little has been measured to mean anything.
    pub fn summary(&self) -> Option<SpeedSummary> {
        let (base_at, base) = self.measured_base?;
        let (last_at, last) = self.latest?;
        let elapsed = last_at.duration_since(base_at);
        if elapsed < MIN_SPAN {
            return None;
        }
        let hashes = last.saturating_sub(base);
        Some(SpeedSummary {
            rate: hashes as f64 / elapsed.as_secs_f64(),
            hashes,
            elapsed,
        })
    }
}

/// Sum of the per-device summaries, for reporting one figure across GPUs.
pub fn combine(summaries: &[Option<SpeedSummary>]) -> Option<SpeedSummary> {
    let present: Vec<SpeedSummary> = summaries.iter().flatten().copied().collect();
    let first = present.first()?;
    Some(SpeedSummary {
        rate: present.iter().map(|s| s.rate).sum(),
        hashes: present.iter().map(|s| s.hashes).sum(),
        // Devices run concurrently for the same wall time, so the longest span
        // describes the measurement rather than the sum of them.
        elapsed: present
            .iter()
            .map(|s| s.elapsed)
            .max()
            .unwrap_or(first.elapsed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn rate_is_measured_over_the_window_not_since_the_start() {
        let start = Instant::now();
        let mut m = SpeedMeter::starting_at(start, Duration::from_secs(3), Duration::ZERO);

        // A very slow first second, then a steady 1000/s for four more.
        m.sample_at(at(start, 0), 0);
        m.sample_at(at(start, 1000), 10);
        for i in 1..=16 {
            m.sample_at(at(start, 1000 + i * 250), 10 + i * 250);
        }

        // Cumulative averaging would report about 802/s here, still dragged
        // down by the slow start five seconds later. The window has moved past
        // it entirely.
        assert!((m.rate() - 1000.0).abs() < 1.0, "rate was {}", m.rate());
    }

    #[test]
    fn a_stalled_device_reads_zero_within_one_window() {
        let start = Instant::now();
        let mut m = SpeedMeter::starting_at(start, Duration::from_secs(3), Duration::ZERO);
        for i in 0..=8 {
            m.sample_at(at(start, i * 250), i * 250);
        }
        assert!(m.rate() > 900.0);

        // Work stops; the counter no longer moves.
        for i in 9..=24 {
            m.sample_at(at(start, i * 250), 2000);
        }
        assert_eq!(m.rate(), 0.0);
    }

    #[test]
    fn summary_excludes_the_warmup() {
        let start = Instant::now();
        let mut m = SpeedMeter::starting_at(start, Duration::from_secs(3), Duration::from_secs(1));

        // 100/s during the warmup second, then 1000/s for two seconds.
        m.sample_at(at(start, 0), 0);
        m.sample_at(at(start, 1000), 100);
        m.sample_at(at(start, 2000), 1100);
        m.sample_at(at(start, 3000), 2100);

        let s = m.summary().expect("warmup has elapsed");
        assert_eq!(s.hashes, 2000);
        assert_eq!(s.elapsed, Duration::from_secs(2));
        assert!((s.rate - 1000.0).abs() < 1.0, "rate was {}", s.rate);
    }

    #[test]
    fn no_summary_before_the_warmup_elapses() {
        let start = Instant::now();
        let mut m = SpeedMeter::starting_at(start, Duration::from_secs(3), Duration::from_secs(10));
        m.sample_at(at(start, 0), 0);
        m.sample_at(at(start, 500), 500);
        assert!(m.summary().is_none());
    }

    #[test]
    fn too_short_a_span_reports_nothing_rather_than_a_huge_number() {
        let start = Instant::now();
        let mut m = SpeedMeter::starting_at(start, Duration::from_secs(3), Duration::ZERO);
        m.sample_at(at(start, 0), 0);
        m.sample_at(at(start, 1), 1_000_000);
        assert_eq!(m.rate(), 0.0);
        assert!(m.summary().is_none());
    }

    #[test]
    fn combine_sums_rates_and_takes_the_longest_span() {
        let a = SpeedSummary {
            rate: 100.0,
            hashes: 200,
            elapsed: Duration::from_secs(2),
        };
        let b = SpeedSummary {
            rate: 50.0,
            hashes: 150,
            elapsed: Duration::from_secs(3),
        };
        let c = combine(&[Some(a), Some(b), None]).unwrap();
        assert_eq!(c.rate, 150.0);
        assert_eq!(c.hashes, 350);
        assert_eq!(c.elapsed, Duration::from_secs(3));
        assert!(combine(&[None, None]).is_none());
    }
}
