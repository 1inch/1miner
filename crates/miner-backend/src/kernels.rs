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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_are_present_and_declare_their_entry_points() {
        assert!(SALT.contains("__kernel void salt_iterate"));
        assert!(PROFANITY.contains("__kernel void profanity_init"));
        for keccak in [KECCAK_TUNED, KECCAK_PLAIN] {
            assert!(keccak.contains("void sha3_keccakf"));
            assert!(keccak.contains("} ethhash;"));
            // Both variants must fold in the trailing keccak pad byte, since
            // callers only supply the leading 0x01 bit.
            assert!(keccak.contains("0x80000000"));
        }
    }
}
