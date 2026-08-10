//! The scoring arguments the kernels read, in the byte layouts they read them.
//!
//! Every kernel binds one of these two shapes: a scoring mode for an ordinary
//! search, or the list of masks `--exact` searches for. All four kernels read
//! these exact bytes, and nothing on the device side would object to a field
//! reordered here — a wrong layout is not a build failure but a mask that
//! quietly matches nothing, or scores against the wrong twenty bytes.
//!
//! `Mode::function` carries a [`miner_core::ScoreFn`] discriminant, which is
//! why that enum's order is fixed by the OpenCL side rather than free to tidy.

use miner_core::ScoreSpec;

/// The scoring mode, matching `mode` in kernels/opencl/salt.cl, the `data1` and
/// `data2` arguments in kernels/opencl/profanity.cl, and `Mode` in
/// kernels/metal/scoring.metal.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mode {
    pub function: u32,
    pub data1: [u8; 20],
    pub data2: [u8; 20],
}

impl From<&ScoreSpec> for Mode {
    fn from(spec: &ScoreSpec) -> Self {
        Self {
            function: spec.function as u32,
            data1: spec.data1,
            data2: spec.data2,
        }
    }
}

/// One `--exact` mask, matching `pattern` in kernels/opencl/salt.cl and
/// `Pattern` in kernels/metal/scoring.metal.
///
/// `mask` has 0xF nibbles where a digit was given and 0 where it was a
/// wildcard, `want` the digits themselves, so a candidate matches when
/// `(address[i] & mask[i]) == want[i]` across all twenty bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pattern {
    pub mask: [u8; 20],
    pub want: [u8; 20],
}

impl From<&ScoreSpec> for Pattern {
    fn from(spec: &ScoreSpec) -> Self {
        Self {
            mask: spec.data1,
            want: spec.data2,
        }
    }
}

/// The masks a job searches for, or an empty list for ordinary scoring.
pub fn patterns(exact: Option<&[ScoreSpec]>) -> Vec<Pattern> {
    exact
        .unwrap_or_default()
        .iter()
        .map(Pattern::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four kernels declare these layouts, and a mismatch is read as garbage
    /// rather than reported as an error.
    #[test]
    fn the_wire_layouts_are_what_the_kernels_declare() {
        assert_eq!(size_of::<Mode>(), 44);
        assert_eq!(size_of::<Pattern>(), 40);
    }

    /// A mask and its wanted digits arrive in `data1` and `data2`, the same two
    /// fields ordinary scoring passes its own operands in. Swapping them here
    /// would match every address with the digits in the wildcard positions.
    #[test]
    fn a_mask_keeps_the_spec_field_order() {
        let spec = ScoreSpec::matching("dead").unwrap();
        let pattern = Pattern::from(&spec);
        assert_eq!(pattern.mask, spec.data1);
        assert_eq!(pattern.want, spec.data2);
    }

    #[test]
    fn a_mode_carries_the_score_function_discriminant() {
        let spec = ScoreSpec::zeros();
        assert_eq!(Mode::from(&spec).function, spec.function as u32);
    }
}
