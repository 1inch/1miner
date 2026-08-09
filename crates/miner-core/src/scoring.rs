//! CPU reference for every scoring function the kernels implement.
//!
//! These are deliberate line-by-line ports of the OpenCL scorers rather than
//! idiomatic rewrites: the GPU and CPU must agree exactly, including the
//! break-on-first-miss behaviour that distinguishes `Leading` from `Range`.

use crate::{Address, CoreError, Result};

/// Mirrors the `ModeFunction` enum shared by eradicate2.cl and eradicate3.cl.
/// Discriminants are the wire format passed to the kernels, so the order is
/// fixed by the OpenCL side and must not be rearranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ScoreFn {
    Benchmark = 0,
    ZeroBytes = 1,
    Matching = 2,
    Leading = 3,
    Range = 4,
    Mirror = 5,
    Doubles = 6,
    LeadingRange = 7,
}

/// A scoring function plus its two 20-byte parameter blocks, laid out exactly
/// as the `mode` struct the kernels read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreSpec {
    pub function: ScoreFn,
    pub data1: [u8; 20],
    pub data2: [u8; 20],
}

impl ScoreSpec {
    fn new(function: ScoreFn) -> Self {
        Self { function, data1: [0; 20], data2: [0; 20] }
    }

    pub fn benchmark() -> Self {
        Self::new(ScoreFn::Benchmark)
    }

    pub fn zero_bytes() -> Self {
        Self::new(ScoreFn::ZeroBytes)
    }

    pub fn mirror() -> Self {
        Self::new(ScoreFn::Mirror)
    }

    pub fn doubles() -> Self {
        Self::new(ScoreFn::Doubles)
    }

    pub fn range(min: u8, max: u8) -> Result<Self> {
        if min > 15 || max > 15 {
            return Err(CoreError::Config(format!(
                "range bounds must be nibbles 0-15, got min={min} max={max}"
            )));
        }
        if min > max {
            return Err(CoreError::Config(format!(
                "range minimum {min} exceeds maximum {max}"
            )));
        }
        let mut s = Self::new(ScoreFn::Range);
        s.data1[0] = min;
        s.data2[0] = max;
        Ok(s)
    }

    pub fn leading_range(min: u8, max: u8) -> Result<Self> {
        let mut s = Self::range(min, max)?;
        s.function = ScoreFn::LeadingRange;
        Ok(s)
    }

    pub fn zeros() -> Self {
        Self::range(0, 0).expect("0..=0 is a valid nibble range")
    }

    pub fn letters() -> Self {
        Self::range(10, 15).expect("10..=15 is a valid nibble range")
    }

    pub fn numbers() -> Self {
        Self::range(0, 9).expect("0..=9 is a valid nibble range")
    }

    pub fn leading(nibble: char) -> Result<Self> {
        let value = nibble.to_digit(16).ok_or_else(|| {
            CoreError::Config(format!("--leading expects one hex digit, got {nibble:?}"))
        })?;
        let mut s = Self::new(ScoreFn::Leading);
        s.data1[0] = value as u8;
        Ok(s)
    }

    /// Left-anchored mask. Non-hex characters are wildcards, so `dead..beef`
    /// and `deadXXbeef` both leave the middle byte unconstrained.
    pub fn matching(pattern: &str) -> Result<Self> {
        Self::mask(pattern, Anchor::Start)
    }

    /// Right-anchored mask. Upstream builds this as a `Matching` spec too; the
    /// kernels have a separate `Trailing` scorer that the host never selects.
    pub fn trailing(pattern: &str) -> Result<Self> {
        Self::mask(pattern, Anchor::End)
    }

    fn mask(pattern: &str, anchor: Anchor) -> Result<Self> {
        let pattern = pattern
            .strip_prefix("0x")
            .or_else(|| pattern.strip_prefix("0X"))
            .unwrap_or(pattern);
        let mut chars: Vec<char> = pattern.chars().collect();
        // A right-anchored pattern aligns to the address's last nibble, so an
        // odd-length one takes a leading wildcard. Chunking from the left and
        // shifting whole bytes would put its final digit in the high nibble of
        // byte 19, moving the whole pattern half a byte towards the front.
        if matches!(anchor, Anchor::End) && chars.len() % 2 == 1 {
            chars.insert(0, '.');
        }
        let bytes = chars.len().div_ceil(2);
        if bytes > 20 {
            return Err(CoreError::Config(format!(
                "pattern covers {bytes} bytes, an address is only 20"
            )));
        }

        let mut s = Self::new(ScoreFn::Matching);
        let offset = match anchor {
            Anchor::Start => 0,
            Anchor::End => 20 - bytes,
        };

        for (index, pair) in chars.chunks(2).enumerate() {
            let (mask_hi, val_hi) = nibble_mask(pair[0]);
            let (mask_lo, val_lo) = pair.get(1).map_or((0, 0), |c| nibble_mask(*c));
            s.data1[offset + index] = (mask_hi << 4) | mask_lo;
            s.data2[offset + index] = (val_hi << 4) | val_lo;
        }
        Ok(s)
    }

    /// How many bytes this specification actually constrains.
    ///
    /// For a mask, that is the maximum score attainable, which is what turns a
    /// best-effort search into an all-or-nothing one: only a candidate scoring
    /// exactly this much satisfies every constrained byte.
    pub fn constrained_bytes(&self) -> u32 {
        match self.function {
            ScoreFn::Matching => self.data1.iter().filter(|m| **m > 0).count() as u32,
            _ => 0,
        }
    }
}

enum Anchor {
    Start,
    End,
}

/// A hex digit contributes a `0xF` mask and its value; anything else is a
/// wildcard contributing neither.
fn nibble_mask(c: char) -> (u8, u8) {
    match c.to_digit(16) {
        Some(v) => (0xF, v as u8),
        None => (0, 0),
    }
}

/// Score a candidate address. Higher is better; the miner keeps the best seen.
pub fn score(spec: &ScoreSpec, address: &Address) -> u32 {
    match spec.function {
        ScoreFn::Benchmark => 0,
        ScoreFn::ZeroBytes => address.iter().filter(|b| **b == 0).count() as u32,
        ScoreFn::Matching => {
            let mut score = 0;
            for (i, byte) in address.iter().enumerate() {
                if spec.data1[i] > 0 && (byte & spec.data1[i]) == spec.data2[i] {
                    score += 1;
                }
            }
            score
        }
        ScoreFn::Leading => {
            let want = spec.data1[0];
            let mut score = 0;
            for byte in address {
                if byte >> 4 != want {
                    break;
                }
                score += 1;
                if byte & 0x0f != want {
                    break;
                }
                score += 1;
            }
            score
        }
        ScoreFn::Range => {
            let (min, max) = (spec.data1[0], spec.data2[0]);
            let mut score = 0;
            for byte in address {
                if (min..=max).contains(&(byte >> 4)) {
                    score += 1;
                }
                if (min..=max).contains(&(byte & 0x0f)) {
                    score += 1;
                }
            }
            score
        }
        ScoreFn::LeadingRange => {
            let (min, max) = (spec.data1[0], spec.data2[0]);
            let mut score = 0;
            for byte in address {
                if !(min..=max).contains(&(byte >> 4)) {
                    break;
                }
                score += 1;
                if !(min..=max).contains(&(byte & 0x0f)) {
                    break;
                }
                score += 1;
            }
            score
        }
        ScoreFn::Mirror => {
            let mut score = 0;
            for i in 0..10 {
                let left = address[9 - i];
                let right = address[10 + i];
                if left & 0x0f != right >> 4 {
                    break;
                }
                score += 1;
                if left >> 4 != right & 0x0f {
                    break;
                }
                score += 1;
            }
            score
        }
        ScoreFn::Doubles => {
            let mut score = 0;
            for byte in address {
                if byte >> 4 != byte & 0x0f {
                    break;
                }
                score += 1;
            }
            score
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexutil::parse_address;

    fn addr(s: &str) -> Address {
        parse_address(s).unwrap()
    }

    /// A 40-digit address with `digits` at the front, zeroes after it.
    fn pad_right(digits: &str) -> String {
        format!("{digits:0<40}")
    }

    /// A 40-digit address with `digits` at the end, zeroes before it.
    fn pad_left(digits: &str) -> String {
        format!("{digits:0>40}")
    }

    #[test]
    fn leading_counts_nibbles_and_stops() {
        let spec = ScoreSpec::leading('0').unwrap();
        assert_eq!(score(&spec, &addr("0x0000012300000000000000000000000000000000")), 5);
        assert_eq!(score(&spec, &addr("0x1000000000000000000000000000000000000000")), 0);
        // A full run of zeroes scores every nibble.
        assert_eq!(score(&spec, &addr("0x0000000000000000000000000000000000000000")), 40);
    }

    #[test]
    fn zeros_counts_anywhere_but_leading_range_stops() {
        let anywhere = ScoreSpec::zeros();
        let leading = ScoreSpec::leading_range(0, 0).unwrap();
        let a = addr("0x0100000000000000000000000000000000000000");
        assert_eq!(score(&anywhere, &a), 39);
        assert_eq!(score(&leading, &a), 1);
    }

    #[test]
    fn zero_bytes_counts_whole_bytes() {
        assert_eq!(
            score(&ScoreSpec::zero_bytes(), &addr("0x0000dead00000000000000000000000000000000")),
            18
        );
    }

    #[test]
    fn letters_and_numbers_partition_the_nibbles() {
        let a = addr("0xabcdef0123456789abcdef0123456789abcdef01");
        assert_eq!(
            score(&ScoreSpec::letters(), &a) + score(&ScoreSpec::numbers(), &a),
            40
        );
    }

    #[test]
    fn matching_is_left_anchored_and_wildcards_are_free() {
        let spec = ScoreSpec::matching("dead").unwrap();
        assert_eq!(score(&spec, &addr("0xdead000000000000000000000000000000000000")), 2);
        assert_eq!(score(&spec, &addr("0xde00000000000000000000000000000000000000")), 1);

        // "de..beef" leaves byte 1 unconstrained, so it contributes nothing.
        let wild = ScoreSpec::matching("de..beef").unwrap();
        assert_eq!(wild.data1[1], 0);
        assert_eq!(score(&wild, &addr("0xdeffbeef00000000000000000000000000000000")), 3);
    }

    #[test]
    fn trailing_is_right_anchored() {
        let spec = ScoreSpec::trailing("beef").unwrap();
        assert_eq!(spec.data1[18], 0xff);
        assert_eq!(spec.data1[19], 0xff);
        assert_eq!(spec.data2[18], 0xbe);
        assert_eq!(spec.data2[19], 0xef);
        // Full match on both anchored bytes.
        assert_eq!(score(&spec, &addr("0x000000000000000000000000000000000000beef")), 2);
        // Only byte 18 lands correctly.
        assert_eq!(score(&spec, &addr("0x000000000000000000000000000000000000beff")), 1);
        // Shifted one byte right: neither anchored byte matches.
        assert_eq!(score(&spec, &addr("0x0000000000000000000000000000000000beefff")), 0);
    }

    /// An odd-length pattern must put its last digit in the *low* nibble of
    /// byte 19, leaving the leading half-byte free. Chunking from the left used
    /// to leave the pattern one nibble too far towards the front, so an address
    /// genuinely ending in "abc" scored zero.
    #[test]
    fn trailing_aligns_an_odd_length_pattern_to_the_last_nibble() {
        let spec = ScoreSpec::trailing("abc").unwrap();
        assert_eq!((spec.data1[18], spec.data2[18]), (0x0f, 0x0a));
        assert_eq!((spec.data1[19], spec.data2[19]), (0xff, 0xbc));
        // The half-masked byte still counts as constrained, so --exact-style
        // all-or-nothing comparisons stay reachable.
        assert_eq!(spec.constrained_bytes(), 2);
        assert_eq!(score(&spec, &addr("0x0000000000000000000000000000000000000abc")), 2);
        // The nibble-shifted address is what this used to search for.
        assert_eq!(score(&spec, &addr("0x000000000000000000000000000000000000abc0")), 0);

        // One digit constrains one nibble of the final byte and nothing else.
        let single = ScoreSpec::trailing("c").unwrap();
        assert_eq!((single.data1[19], single.data2[19]), (0x0f, 0x0c));
        assert_eq!(single.constrained_bytes(), 1);
        assert_eq!(score(&single, &addr("0x000000000000000000000000000000000000000c")), 1);
        assert_eq!(score(&single, &addr("0x00000000000000000000000000000000000000c0")), 0);
    }

    /// Both anchors have to hold at every pattern length, not only the even
    /// ones the fixed-string tests above happen to use.
    #[test]
    fn mask_alignment_holds_at_every_pattern_length() {
        const DIGITS: &str = "123456789abcdef";

        for len in 1..=DIGITS.len() {
            let head = &DIGITS[..len];
            let tail = &DIGITS[DIGITS.len() - len..];
            let start = ScoreSpec::matching(head).unwrap();
            let end = ScoreSpec::trailing(tail).unwrap();

            // At its anchor the pattern satisfies every byte it constrains.
            let (from_start, from_end) = (pad_right(head), pad_left(tail));
            assert_eq!(
                score(&start, &addr(&from_start)),
                start.constrained_bytes(),
                "--matching {head} should match {from_start} in full"
            );
            assert_eq!(
                score(&end, &addr(&from_end)),
                end.constrained_bytes(),
                "--trailing {tail} should match {from_end} in full"
            );

            // One nibble away from it, it no longer does.
            let off_start = pad_right(&format!("0{head}"));
            let off_end = pad_left(&format!("{tail}0"));
            assert!(
                score(&start, &addr(&off_start)) < start.constrained_bytes(),
                "--matching {head} should not match {off_start} in full"
            );
            assert!(
                score(&end, &addr(&off_end)) < end.constrained_bytes(),
                "--trailing {tail} should not match {off_end} in full"
            );
        }
    }

    /// The wildcard pad must not cost a right-anchored pattern its last byte:
    /// 39 digits still cover an address, 41 still do not.
    #[test]
    fn the_wildcard_pad_does_not_move_the_length_limit() {
        let widest = ScoreSpec::trailing(&"a".repeat(39)).unwrap();
        assert_eq!(widest.constrained_bytes(), 20);
        assert_eq!(widest.data1[0], 0x0f);
        assert!(ScoreSpec::trailing(&"a".repeat(40)).is_ok());
        assert!(ScoreSpec::trailing(&"a".repeat(41)).is_err());
        assert!(ScoreSpec::matching(&"a".repeat(41)).is_err());
    }

    #[test]
    fn mirror_reflects_around_the_centre() {
        // Bytes 9 and 10 are 0x12 and 0x21, so two nibbles mirror; bytes 8 and
        // 11 (0x34 vs 0x00) then disagree and stop the count.
        assert_eq!(
            score(&ScoreSpec::mirror(), &addr("0x0000000000000000341221000000000000000000")),
            2
        );
        // An all-zero address mirrors completely.
        assert_eq!(
            score(&ScoreSpec::mirror(), &addr("0x0000000000000000000000000000000000000000")),
            20
        );
        assert_eq!(
            score(&ScoreSpec::mirror(), &addr("0x0000000000000000001200000000000000000000")),
            0
        );
    }

    #[test]
    fn doubles_requires_matching_nibble_pairs() {
        // 00 aa 11 bb are pairs, 0x12 is not, so the run stops at four.
        assert_eq!(
            score(&ScoreSpec::doubles(), &addr("0x00aa11bb12000000000000000000000000000000")),
            4
        );
        // A leading 0x01 is not a pair, so nothing counts.
        assert_eq!(
            score(&ScoreSpec::doubles(), &addr("0x0102000000000000000000000000000000000000")),
            0
        );
        // Trailing zero bytes are pairs too, so an all-zero address scores 20.
        assert_eq!(
            score(&ScoreSpec::doubles(), &addr("0x0000000000000000000000000000000000000000")),
            20
        );
    }

    #[test]
    fn benchmark_never_scores() {
        assert_eq!(
            score(&ScoreSpec::benchmark(), &addr("0x0000000000000000000000000000000000000000")),
            0
        );
    }

    #[test]
    fn range_rejects_invalid_bounds() {
        assert!(ScoreSpec::range(0, 16).is_err());
        assert!(ScoreSpec::range(9, 3).is_err());
        assert!(ScoreSpec::leading('g').is_err());
        assert!(ScoreSpec::matching(&"a".repeat(42)).is_err());
    }

    /// The count drives --exact: a candidate has to reach it to satisfy every
    /// constrained byte, so wildcards must not inflate it.
    #[test]
    fn constrained_bytes_counts_only_masked_positions() {
        assert_eq!(ScoreSpec::matching("dead").unwrap().constrained_bytes(), 2);
        assert_eq!(ScoreSpec::trailing("beef").unwrap().constrained_bytes(), 2);
        // A half-masked byte constrains that byte too: "abc" pads to ".abc".
        assert_eq!(ScoreSpec::trailing("abc").unwrap().constrained_bytes(), 2);
        // "de..beef" leaves byte 1 free, so three bytes are constrained.
        assert_eq!(ScoreSpec::matching("de..beef").unwrap().constrained_bytes(), 3);
        // A whole address is 20 bytes.
        assert_eq!(ScoreSpec::matching(&"a".repeat(40)).unwrap().constrained_bytes(), 20);
        // Only masks constrain individual bytes.
        assert_eq!(ScoreSpec::zeros().constrained_bytes(), 0);
    }

    /// An address satisfying every constrained byte must score exactly the
    /// count, and one that misses any of them must score less.
    #[test]
    fn full_mask_match_scores_the_constrained_count() {
        let spec = ScoreSpec::matching("de..beef").unwrap();
        let want = spec.constrained_bytes();
        assert_eq!(score(&spec, &addr("0xde99beef00000000000000000000000000000000")), want);
        assert!(score(&spec, &addr("0xde99beee00000000000000000000000000000000")) < want);
    }
}
