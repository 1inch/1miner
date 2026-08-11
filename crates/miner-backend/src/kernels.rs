//! Kernel sources, embedded at compile time.
//!
//! profanity2 and ERADICATE2/3 read their `.cl` files from the working
//! directory at runtime, which makes a container or a moved binary fail in
//! confusing ways. Embedding removes that failure mode entirely.

/// ERADICATE2/3's tuned Keccak-f permutation.
pub const KECCAK_TUNED: &str = include_str!("../../../kernels/opencl/keccak_tuned.cl");

/// profanity2's Keccak-f permutation.
pub const KECCAK_PLAIN: &str = include_str!("../../../kernels/opencl/keccak_plain.cl");

/// Unified create2 / create3 / 1nft salt search.
pub const SALT: &str = include_str!("../../../kernels/opencl/salt.cl");

/// secp256k1 vanity search over a seed public key.
pub const PROFANITY: &str = include_str!("../../../kernels/opencl/profanity.cl");

/// Keccak-f for Metal, prepended to every Metal library. It carries the
/// `#include <metal_stdlib>` both kernels below need, so it stays first.
pub const METAL_KECCAK: &str = include_str!("../../../kernels/metal/keccak.metal");

/// Scoring for Metal, shared by the salt and profanity kernels. OpenCL keeps a
/// copy per program; one Metal library can hold one.
pub const METAL_SCORING: &str = include_str!("../../../kernels/metal/scoring.metal");

/// Metal create2 / create3 / 1nft salt search.
pub const METAL_SALT: &str = include_str!("../../../kernels/metal/salt.metal");

/// Metal secp256k1 vanity search over a seed public key.
pub const METAL_PROFANITY: &str = include_str!("../../../kernels/metal/profanity.metal");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_are_present_and_declare_their_entry_points() {
        assert!(SALT.contains("__kernel void salt_iterate"));
        assert!(PROFANITY.contains("__kernel void profanity_init"));
        assert!(METAL_SALT.contains("kernel void salt_iterate"));
        assert!(METAL_PROFANITY.contains("kernel void profanity_init"));
        assert!(METAL_KECCAK.contains("void keccakf(thread ulong* st)"));
        assert!(METAL_KECCAK.contains("void keccakf_address(thread ulong* st)"));
        assert!(METAL_SCORING.contains("int score_address"));
        for keccak in [KECCAK_TUNED, KECCAK_PLAIN] {
            assert!(keccak.contains("void sha3_keccakf(ethhash"));
            // The salt kernel calls the address-only permutation, and --kernel
            // chooses the variant at run time, so a variant that does not
            // declare it fails to build on the device the first time somebody
            // selects it rather than here.
            assert!(keccak.contains("void sha3_keccakf_address(ethhash"));
            assert!(keccak.contains("} ethhash;"));
            // Both variants must fold in the trailing keccak pad byte, since
            // callers only supply the leading 0x01 bit.
            assert!(keccak.contains("0x80000000"));
        }
        // As must the Metal port, whose callers make the same assumption.
        assert!(METAL_KECCAK.contains("st[16] ^= 0x8000000000000000UL;"));
    }

    /// A Metal library is one translation unit built from concatenated
    /// sources, so the `#include` may appear once and has to come first.
    /// Duplicating it in a later part fails the build with a message that
    /// names a line number in a file nobody wrote.
    #[test]
    fn only_the_metal_prelude_includes_the_standard_library() {
        let directives = |source: &str| {
            source
                .lines()
                .filter(|line| {
                    let line = line.trim_start();
                    line.starts_with("#include") || line.starts_with("using namespace")
                })
                .count()
        };
        assert_eq!(directives(METAL_KECCAK), 2);
        for part in [METAL_SCORING, METAL_SALT, METAL_PROFANITY] {
            assert_eq!(directives(part), 0);
        }
    }

    /// The result counter is a `uint`, and a narrower copy of it would make
    /// every 256th writer believe it was first. Only a GPU shows what that
    /// costs, so this is the tripwire for a machine without one.
    #[test]
    fn the_first_writer_check_reads_the_whole_counter() {
        for source in [SALT, PROFANITY] {
            assert!(source.contains("const uint hasResult = atomic_inc"));
            assert!(!source.contains("uchar hasResult"));
        }
    }

    /// All three scorers test for a byte's two nibbles being equal by XORing it
    /// with itself shifted down and looking at the low four bits, which is one
    /// instruction rather than two masks and a compare.
    ///
    /// The identity is checked here over every byte there is, because it is the
    /// sort of trick that is either right for all 256 values or wrong in a way
    /// no realistic search would run into: the device-side test compares only
    /// the handful of hits that clear a climbing bar, and a scorer that
    /// undercounts by one nibble can survive it.
    #[test]
    fn equal_nibbles_are_what_the_scorers_test_for() {
        for byte in 0u8..=u8::MAX {
            let trick = ((byte >> 4) ^ byte) & 0x0f == 0;
            let plain = (byte & 0xF0) >> 4 == byte & 0x0F;
            assert_eq!(trick, plain, "the two forms disagree for {byte:#04x}");
        }

        for (source, form) in [
            (PROFANITY, "(((byte >> 4) ^ byte) & 0x0f) == 0"),
            (SALT, "(((hash[i] >> 4) ^ hash[i]) & 0x0f) == 0"),
            (METAL_SCORING, "(((hash[i] >> 4) ^ hash[i]) & 0x0f) != 0"),
        ] {
            assert!(source.contains(form), "a scorer no longer reads {form}");
        }
    }

    /// `bswap32` uses its argument twice inside `&` expressions, so both uses
    /// have to be bracketed. Every call site passes a plain array element
    /// today, which is why an argument like `a | b` would go wrong quietly.
    #[test]
    fn the_byte_swap_macro_brackets_its_argument() {
        let definition = PROFANITY
            .lines()
            .find(|line| line.starts_with("#define bswap32"))
            .expect("profanity.cl defines bswap32");
        assert!(
            definition.contains("rotate((n) & ") && definition.contains("rotate((n), "),
            "unbracketed macro argument: {definition}"
        );
    }
}
