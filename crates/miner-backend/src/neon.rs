//! Two-way SIMD Keccak for the CPU backend on aarch64.
//!
//! Each of the 25 Keccak lanes is held as a `uint64x2_t`, so one permutation
//! processes two independent candidates. NEON is mandatory on aarch64, so no
//! runtime detection is needed for it; the ARMv8.2 SHA3 extension is not, and
//! [`keccak_f_x2_sha3`] is used in place of [`keccak_f_x2`] where it is present.
//!
//! The permutation follows the same convention as the OpenCL and Metal kernels:
//! callers set the leading `0x01` pad bit and the permutation applies the
//! trailing `0x80` itself, so it is valid for single-block messages only. That
//! is all the 85-byte and 23-byte pre-images need. Tests assert the output
//! matches the `sha3`-crate reference, which arrives at the same answer by
//! doing full standard padding instead.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use miner_core::{Address, SaltConfig, mode::STATE_WORDS};

/// Candidates processed per permutation.
pub const LANES: usize = 2;

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotate both lanes left. The amount is a literal so the shifts stay
/// immediate; `vshlq_n_u64` requires a constant anyway.
macro_rules! rotl {
    ($x:expr, $n:literal) => {
        vorrq_u64(vshlq_n_u64::<$n>($x), vshrq_n_u64::<{ 64 - $n }>($x))
    };
}

/// `a ^ (!b & c)`, the Keccak chi step, as one XOR and one BIC.
///
/// # Safety
///
/// Requires NEON, which is part of the aarch64 baseline this module is gated on.
#[inline(always)]
unsafe fn chi(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    // SAFETY: both intrinsics are register-only, so the target feature is the
    // whole obligation and aarch64 always has it.
    unsafe { veorq_u64(a, vbicq_u64(c, b)) }
}

/// Keccak-f[1600] over two states at once.
///
/// State index `i` corresponds to Keccak lane `A[x, y]` with `i = x + 5y`, the
/// same layout the OpenCL kernel uses. Every index below is a literal so the
/// array stays in registers; computed indices spill and roughly halve the rate,
/// which is the mistake the Metal kernel made first.
///
/// # Safety
///
/// Requires NEON, which is part of the aarch64 baseline this module is gated on.
#[inline]
unsafe fn keccak_f_x2(st: &mut [uint64x2_t; 25]) {
    // SAFETY: every intrinsic in this block is register-only — no loads, no
    // stores, no pointers — so NEON's availability is the only obligation, and
    // aarch64 always has it. Indices are literals bounded by the array's 25.
    unsafe {
        // Trailing keccak pad byte: byte 135 is the top byte of lane 16.
        st[16] = veorq_u64(st[16], vdupq_n_u64(0x8000_0000_0000_0000));

        for rc in ROUND_CONSTANTS {
            // Theta.
            let c0 = veorq_u64(
                veorq_u64(veorq_u64(st[0], st[5]), veorq_u64(st[10], st[15])),
                st[20],
            );
            let c1 = veorq_u64(
                veorq_u64(veorq_u64(st[1], st[6]), veorq_u64(st[11], st[16])),
                st[21],
            );
            let c2 = veorq_u64(
                veorq_u64(veorq_u64(st[2], st[7]), veorq_u64(st[12], st[17])),
                st[22],
            );
            let c3 = veorq_u64(
                veorq_u64(veorq_u64(st[3], st[8]), veorq_u64(st[13], st[18])),
                st[23],
            );
            let c4 = veorq_u64(
                veorq_u64(veorq_u64(st[4], st[9]), veorq_u64(st[14], st[19])),
                st[24],
            );

            let d4 = veorq_u64(rotl!(c0, 1), c3);
            let d0 = veorq_u64(rotl!(c2, 1), c0);
            let d1 = veorq_u64(rotl!(c4, 1), c2);
            let d2 = veorq_u64(rotl!(c1, 1), c4);
            let d3 = veorq_u64(rotl!(c3, 1), c1);

            st[0] = veorq_u64(st[0], d2);
            st[5] = veorq_u64(st[5], d2);
            st[10] = veorq_u64(st[10], d2);
            st[15] = veorq_u64(st[15], d2);
            st[20] = veorq_u64(st[20], d2);

            st[1] = veorq_u64(st[1], d0);
            st[6] = veorq_u64(st[6], d0);
            st[11] = veorq_u64(st[11], d0);
            st[16] = veorq_u64(st[16], d0);
            st[21] = veorq_u64(st[21], d0);

            st[2] = veorq_u64(st[2], d3);
            st[7] = veorq_u64(st[7], d3);
            st[12] = veorq_u64(st[12], d3);
            st[17] = veorq_u64(st[17], d3);
            st[22] = veorq_u64(st[22], d3);

            st[3] = veorq_u64(st[3], d1);
            st[8] = veorq_u64(st[8], d1);
            st[13] = veorq_u64(st[13], d1);
            st[18] = veorq_u64(st[18], d1);
            st[23] = veorq_u64(st[23], d1);

            st[4] = veorq_u64(st[4], d4);
            st[9] = veorq_u64(st[9], d4);
            st[14] = veorq_u64(st[14], d4);
            st[19] = veorq_u64(st[19], d4);
            st[24] = veorq_u64(st[24], d4);

            // Rho and pi as a single 24-element rotation chain.
            let t = rotl!(st[1], 1);
            st[1] = rotl!(st[6], 44);
            st[6] = rotl!(st[9], 20);
            st[9] = rotl!(st[22], 61);
            st[22] = rotl!(st[14], 39);
            st[14] = rotl!(st[20], 18);
            st[20] = rotl!(st[2], 62);
            st[2] = rotl!(st[12], 43);
            st[12] = rotl!(st[13], 25);
            st[13] = rotl!(st[19], 8);
            st[19] = rotl!(st[23], 56);
            st[23] = rotl!(st[15], 41);
            st[15] = rotl!(st[4], 27);
            st[4] = rotl!(st[24], 14);
            st[24] = rotl!(st[21], 2);
            st[21] = rotl!(st[8], 55);
            st[8] = rotl!(st[16], 45);
            st[16] = rotl!(st[5], 36);
            st[5] = rotl!(st[3], 28);
            st[3] = rotl!(st[18], 21);
            st[18] = rotl!(st[17], 15);
            st[17] = rotl!(st[11], 10);
            st[11] = rotl!(st[7], 6);
            st[7] = rotl!(st[10], 3);
            st[10] = t;

            // Chi, one row of five at a time.
            for row in 0..5 {
                let base = row * 5;
                let a0 = st[base];
                let a1 = st[base + 1];
                st[base] = chi(a0, a1, st[base + 2]);
                st[base + 1] = chi(a1, st[base + 2], st[base + 3]);
                st[base + 2] = chi(st[base + 2], st[base + 3], st[base + 4]);
                st[base + 3] = chi(st[base + 3], st[base + 4], a0);
                st[base + 4] = chi(st[base + 4], a0, a1);
            }

            // Iota.
            st[0] = veorq_u64(st[0], vdupq_n_u64(rc));
        }
    }
}

/// Keccak-f[1600] over two states at once, using the ARMv8.2 SHA3 extension.
///
/// The same permutation as [`keccak_f_x2`], lane for lane, in the four
/// instructions that exist for it: `eor3` folds three XORs into one, `rax1` does
/// theta's rotate-and-XOR, `xar` does theta's XOR and rho's rotation together,
/// and `bcax` does chi's `a ^ (!b & c)`. That leaves the round body at roughly
/// a third of the operations.
///
/// `xar` is why the rho chain below carries a `d` value per step: the XOR that
/// the plain version applies to all 25 lanes first is folded into each rotation
/// instead. Which `d` a lane takes follows its column, exactly as the plain
/// version's five blocks of five do. `st[0]` is the one lane rho does not
/// rotate, so it keeps a plain XOR — `xar` cannot express a rotation of zero,
/// its immediate stopping at 63.
///
/// # Safety
///
/// Requires the `sha3` target feature, which is not part of the aarch64
/// baseline. Callers must have checked for it; [`sha3_enabled`] is that check.
#[target_feature(enable = "sha3")]
unsafe fn keccak_f_x2_sha3(st: &mut [uint64x2_t; 25]) {
    /// `rotl(a ^ b, N)`, which `xar` computes as a rotate right by `64 - N`.
    macro_rules! xar {
        ($a:expr, $b:expr, $n:literal) => {
            vxarq_u64::<{ 64 - $n }>($a, $b)
        };
    }

    // Every intrinsic here is register-only, so the target features are the
    // whole obligation and this function's own attribute carries them: inside
    // it the calls need no unsafe block. Indices are literals bounded by 25.
    {
        // Trailing keccak pad byte: byte 135 is the top byte of lane 16.
        st[16] = veorq_u64(st[16], vdupq_n_u64(0x8000_0000_0000_0000));

        for rc in ROUND_CONSTANTS {
            // Theta's column parities, two three-way XORs each.
            let c0 = veor3q_u64(veor3q_u64(st[0], st[5], st[10]), st[15], st[20]);
            let c1 = veor3q_u64(veor3q_u64(st[1], st[6], st[11]), st[16], st[21]);
            let c2 = veor3q_u64(veor3q_u64(st[2], st[7], st[12]), st[17], st[22]);
            let c3 = veor3q_u64(veor3q_u64(st[3], st[8], st[13]), st[18], st[23]);
            let c4 = veor3q_u64(veor3q_u64(st[4], st[9], st[14]), st[19], st[24]);

            // `rax1(a, b)` is `a ^ rotl(b, 1)`, which is what each of these is.
            let d4 = vrax1q_u64(c3, c0);
            let d0 = vrax1q_u64(c0, c2);
            let d1 = vrax1q_u64(c2, c4);
            let d2 = vrax1q_u64(c4, c1);
            let d3 = vrax1q_u64(c1, c3);

            // Rho and pi as the same 24-element rotation chain the plain
            // version uses, with theta's XOR folded into every step.
            let t = xar!(st[1], d0, 1);
            st[1] = xar!(st[6], d0, 44);
            st[6] = xar!(st[9], d4, 20);
            st[9] = xar!(st[22], d3, 61);
            st[22] = xar!(st[14], d4, 39);
            st[14] = xar!(st[20], d2, 18);
            st[20] = xar!(st[2], d3, 62);
            st[2] = xar!(st[12], d3, 43);
            st[12] = xar!(st[13], d1, 25);
            st[13] = xar!(st[19], d4, 8);
            st[19] = xar!(st[23], d1, 56);
            st[23] = xar!(st[15], d2, 41);
            st[15] = xar!(st[4], d4, 27);
            st[4] = xar!(st[24], d4, 14);
            st[24] = xar!(st[21], d0, 2);
            st[21] = xar!(st[8], d1, 55);
            st[8] = xar!(st[16], d0, 45);
            st[16] = xar!(st[5], d2, 36);
            st[5] = xar!(st[3], d1, 28);
            st[3] = xar!(st[18], d1, 21);
            st[18] = xar!(st[17], d3, 15);
            st[17] = xar!(st[11], d0, 10);
            st[11] = xar!(st[7], d3, 6);
            st[7] = xar!(st[10], d2, 3);
            st[10] = t;
            st[0] = veorq_u64(st[0], d2);

            // Chi, one row of five at a time. `bcax(a, c, b)` is `a ^ (c & !b)`.
            for row in 0..5 {
                let base = row * 5;
                let a0 = st[base];
                let a1 = st[base + 1];
                st[base] = vbcaxq_u64(a0, st[base + 2], a1);
                st[base + 1] = vbcaxq_u64(a1, st[base + 3], st[base + 2]);
                st[base + 2] = vbcaxq_u64(st[base + 2], st[base + 4], st[base + 3]);
                st[base + 3] = vbcaxq_u64(st[base + 3], a0, st[base + 4]);
                st[base + 4] = vbcaxq_u64(st[base + 4], a1, a0);
            }

            // Iota.
            st[0] = veorq_u64(st[0], vdupq_n_u64(rc));
        }
    }
}

/// Whether the SHA3 extension's permutation is used.
///
/// Not part of the aarch64 baseline, unlike NEON, so it is detected rather than
/// assumed; every Apple silicon chip has it. `MINER_NO_SHA3=1` forces the plain
/// NEON path, both to A/B the two and as a way to keep mining if the extension
/// path ever misbehaves on some hardware.
fn sha3_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("MINER_NO_SHA3").as_deref(),
            Ok("1") | Ok("true")
        ) && std::arch::is_aarch64_feature_detected!("sha3")
    })
}

/// Whichever permutation this machine can run. Both are checked against each
/// other in tests, so which one runs stays a performance choice.
///
/// # Safety
///
/// Requires NEON, which is part of the aarch64 baseline this module is gated on.
#[inline]
unsafe fn permute(st: &mut [uint64x2_t; 25]) {
    if sha3_enabled() {
        // SAFETY: `sha3_enabled` is true only when the feature was detected.
        unsafe { keccak_f_x2_sha3(st) }
    } else {
        // SAFETY: NEON alone, which aarch64 always has.
        unsafe { keccak_f_x2(st) }
    }
}

/// Pack two scalar states into lane-interleaved vectors.
///
/// # Safety
///
/// Requires NEON, which is part of the aarch64 baseline this module is gated on.
#[inline]
unsafe fn pack(a: &[u64; 25], b: &[u64; 25]) -> [uint64x2_t; 25] {
    // SAFETY: register-only, so NEON's availability is the only obligation.
    let mut out = [unsafe { vdupq_n_u64(0) }; 25];
    for (i, slot) in out.iter_mut().enumerate() {
        let pair = [a[i], b[i]];
        // SAFETY: the load reads exactly the two `u64` of `pair`, a live local
        // array, and an aligned `[u64; 2]` is a valid source for `vld1q_u64`.
        *slot = unsafe { vld1q_u64(pair.as_ptr()) };
    }
    out
}

/// Extract the 20-byte address at bytes 12..32 of each lane's state.
///
/// Expanding all 25 lanes to take twenty bytes out of each looks like 360 bytes
/// of copying nothing reads, since the address can only come from lanes 1 to 4.
/// Reading just those four costs 15% — measured, on top of the state change
/// above — because the state is still live in the permutation's loop and
/// reaching into part of it is enough to put the whole array on the stack. This
/// is the same trap the comment on `keccak_f_x2` describes.
///
/// # Safety
///
/// Requires NEON, which is part of the aarch64 baseline this module is gated on.
#[inline]
unsafe fn unpack_addresses(st: &[uint64x2_t; 25]) -> [Address; LANES] {
    let mut bytes = [[0u8; 200]; LANES];
    for (i, lane) in st.iter().enumerate() {
        let mut pair = [0u64; 2];
        // SAFETY: the store writes exactly the two `u64` of `pair`, a live
        // local array with room for both lanes.
        unsafe { vst1q_u64(pair.as_mut_ptr(), *lane) };
        for (slot, word) in bytes.iter_mut().zip(pair) {
            slot[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }
    }

    let mut out = [[0u8; 20]; LANES];
    for (addr, state) in out.iter_mut().zip(&bytes) {
        addr.copy_from_slice(&state[12..32]);
    }
    out
}

/// The 200-byte keccak state for one work item, as 25 lanes.
///
/// Rebuilding the pre-image per candidate reads like waste, since only three
/// words differ between work items and the kernels bump exactly those. Building
/// the base state once per run and patching those words was measured and is
/// slower: 174 against 156 MH/s on an M4 Max, create2, alternating order with
/// the change and without, and no better with the helper marked `#[inline]`.
/// LLVM already lifts everything here that does not depend on the work item out
/// of the candidate loop, and giving it a prepared state to read from memory
/// takes that away. Leave it alone.
fn state_for(cfg: &SaltConfig, device_index: u32, global_id: u32, round: u32) -> [u64; 25] {
    let salt = cfg.salt_at(device_index, global_id, round);
    let mut bytes = cfg.state();
    bytes[21..53].copy_from_slice(&salt);

    let mut words = [0u64; 25];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    }
    words
}

/// The 23-byte `rlp([proxy, 1])` pre-image for the CREATE3 second hash.
fn create_state_for(proxy: &Address) -> [u64; 25] {
    let mut bytes = [0u8; 200];
    bytes[0] = 0xd6;
    bytes[1] = 0x94;
    bytes[2..22].copy_from_slice(proxy);
    bytes[22] = 0x01;
    // Leading keccak pad bit; the permutation adds the trailing 0x80.
    bytes[23] ^= 0x01;

    let mut words = [0u64; 25];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    }
    words
}

/// Derive two addresses at once for two work items of the same job.
pub fn addresses(
    cfg: &SaltConfig,
    device_index: u32,
    global_ids: [u32; LANES],
    round: u32,
) -> [Address; LANES] {
    debug_assert_eq!(STATE_WORDS, 50, "state layout changed");

    let a = state_for(cfg, device_index, global_ids[0], round);
    let b = state_for(cfg, device_index, global_ids[1], round);

    // SAFETY: NEON is guaranteed present on aarch64, and every pointer used
    // below refers to a local fixed-size array.
    let first = unsafe {
        let mut st = pack(&a, &b);
        permute(&mut st);
        unpack_addresses(&st)
    };

    if !cfg.mode.needs_second_hash() {
        return first;
    }

    // SAFETY: as for the first hash above — NEON is guaranteed on aarch64, and
    // the only pointers involved are to local fixed-size arrays.
    unsafe {
        let a = create_state_for(&first[0]);
        let b = create_state_for(&first[1]);
        let mut st = pack(&a, &b);
        permute(&mut st);
        unpack_addresses(&st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miner_core::{DEFAULT_PROXY_CODE_HASH, MineMode, parse_address};

    fn config(mode: MineMode) -> SaltConfig {
        let deployer = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let caller = (mode == MineMode::Nft)
            .then(|| parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap());
        let mut base = [0u8; 32];
        for (i, b) in base.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(23).wrapping_add(3);
        }
        SaltConfig::new(mode, deployer, DEFAULT_PROXY_CODE_HASH, base, caller).unwrap()
    }

    /// The whole point of this module: two lanes, computed by a hand-written
    /// permutation, must equal the `sha3`-crate reference for every mode. The
    /// two implementations pad differently, so agreement is meaningful.
    #[test]
    fn neon_matches_the_scalar_reference() {
        for mode in [MineMode::Create2, MineMode::Create3, MineMode::Nft] {
            let cfg = config(mode);
            for round in [1u32, 7, 4096] {
                for gid in [0u32, 1, 2, 12_345, u32::MAX - 1] {
                    let pair = [gid, gid.wrapping_add(1)];
                    let got = addresses(&cfg, 0, pair, round);
                    for (lane, id) in pair.iter().enumerate() {
                        let want = cfg.address_for_salt(&cfg.salt_at(0, *id, round));
                        assert_eq!(
                            got[lane],
                            want,
                            "{} mismatch at gid {id} round {round}",
                            mode.as_str()
                        );
                    }
                }
            }
        }
    }

    /// Both lanes of a packed state, so two permutations can be compared.
    fn lanes(st: &[uint64x2_t; 25]) -> [[u64; 2]; 25] {
        let mut out = [[0u64; 2]; 25];
        for (slot, lane) in out.iter_mut().zip(st) {
            // SAFETY: the store writes exactly the two `u64` of `slot`.
            unsafe { vst1q_u64(slot.as_mut_ptr(), *lane) };
        }
        out
    }

    /// The SHA3-extension permutation has to agree with the plain one on every
    /// lane of every state, since which of the two runs is a property of the
    /// machine rather than of the search. This is what catches a mistranscribed
    /// step of the rho chain, where the folded theta XOR makes each line carry
    /// one more thing to get wrong.
    #[test]
    fn the_sha3_permutation_matches_the_plain_one() {
        if !std::arch::is_aarch64_feature_detected!("sha3") {
            eprintln!("skipping SHA3 comparison: the extension is absent");
            return;
        }

        // Something structured, something sparse and something dense, since a
        // wrong lane can hide behind a state that is mostly one value.
        let states: [[u64; 25]; 4] = [
            [0; 25],
            std::array::from_fn(|i| i as u64),
            std::array::from_fn(|i| 0x0123_4567_89ab_cdefu64.wrapping_mul(i as u64 + 1)),
            std::array::from_fn(|i| !(1u64 << (i % 64))),
        ];

        for a in &states {
            for b in &states {
                // SAFETY: NEON is baseline, and the extension was detected
                // above; every pointer is to a live local array.
                let (plain, extended) = unsafe {
                    let mut p = pack(a, b);
                    keccak_f_x2(&mut p);
                    let mut e = pack(a, b);
                    keccak_f_x2_sha3(&mut e);
                    (lanes(&p), lanes(&e))
                };
                assert_eq!(plain, extended, "permutations disagree");
            }
        }
    }

    /// The two lanes must stay independent; a bug that broadcasts one lane over
    /// the other would still match the reference for identical inputs.
    #[test]
    fn lanes_are_independent() {
        let cfg = config(MineMode::Create3);
        let got = addresses(&cfg, 0, [11, 22], 3);
        assert_ne!(got[0], got[1]);
        assert_eq!(got[0], cfg.address_for_salt(&cfg.salt_at(0, 11, 3)));
        assert_eq!(got[1], cfg.address_for_salt(&cfg.salt_at(0, 22, 3)));

        // Swapping the inputs must swap the outputs.
        let swapped = addresses(&cfg, 0, [22, 11], 3);
        assert_eq!(swapped[0], got[1]);
        assert_eq!(swapped[1], got[0]);
    }
}
