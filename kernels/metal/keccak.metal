/* Keccak-f[1600] for the Metal backend, a direct port of the tuned OpenCL
 * permutation in kernels/opencl/keccak_tuned.cl.
 *
 * Prepended to every Metal library this project builds, the way the OpenCL
 * backend prepends keccak_tuned.cl to salt.cl and profanity.cl. It therefore
 * carries the `#include` both kernels need and must stay first in the
 * concatenation.
 */

#include <metal_stdlib>
using namespace metal;

constant ulong kRoundConstants[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

static inline ulong rotl(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

/* Every state index below is a literal, which lets the compiler keep all 25
 * lanes in registers. An earlier version used computed indices into a scratch
 * array and ran at roughly half this speed because the array spilled to
 * memory.
 *
 * The `^= 0x80 << 56` on lane 16 is the trailing keccak pad byte (byte 135 of
 * the rate), matching `h->d[33] ^= 0x80000000` in the OpenCL implementation.
 * Callers therefore only set the leading 0x01 bit. This makes the function
 * valid for single-block messages, which is all the 85-byte, 64-byte and
 * 23-byte pre-images here need.
 */
#define TH_ELT_SHORT(t, d, c) t = rotl(d, 1) ^ c

#define THETA(s00, s01, s02, s03, s04,                         \
              s10, s11, s12, s13, s14,                         \
              s20, s21, s22, s23, s24,                         \
              s30, s31, s32, s33, s34,                         \
              s40, s41, s42, s43, s44)                         \
{                                                              \
    t0 = s00 ^ s01 ^ s02 ^ s03 ^ s04;                          \
    t1 = s10 ^ s11 ^ s12 ^ s13 ^ s14;                          \
    t2 = s20 ^ s21 ^ s22 ^ s23 ^ s24;                          \
    t3 = s30 ^ s31 ^ s32 ^ s33 ^ s34;                          \
    t4 = s40 ^ s41 ^ s42 ^ s43 ^ s44;                          \
                                                               \
    TH_ELT_SHORT(t5, t0, t3);                                  \
    TH_ELT_SHORT(t0, t2, t0);                                  \
    TH_ELT_SHORT(t2, t4, t2);                                  \
    TH_ELT_SHORT(t4, t1, t4);                                  \
    TH_ELT_SHORT(t1, t3, t1);                                  \
                                                               \
    s00 ^= t4; s01 ^= t4; s02 ^= t4; s03 ^= t4; s04 ^= t4;     \
    s10 ^= t0; s11 ^= t0; s12 ^= t0; s13 ^= t0; s14 ^= t0;     \
    s20 ^= t1; s21 ^= t1; s22 ^= t1; s23 ^= t1; s24 ^= t1;     \
    s30 ^= t2; s31 ^= t2; s32 ^= t2; s33 ^= t2; s34 ^= t2;     \
    s40 ^= t5; s41 ^= t5; s42 ^= t5; s43 ^= t5; s44 ^= t5;     \
}

#define RHOPI(s00, s01, s02, s03, s04,                         \
              s10, s11, s12, s13, s14,                         \
              s20, s21, s22, s23, s24,                         \
              s30, s31, s32, s33, s34,                         \
              s40, s41, s42, s43, s44)                         \
{                                                              \
    t0  = rotl(s10,  1);                                       \
    s10 = rotl(s11, 44);                                       \
    s11 = rotl(s41, 20);                                       \
    s41 = rotl(s24, 61);                                       \
    s24 = rotl(s42, 39);                                       \
    s42 = rotl(s04, 18);                                       \
    s04 = rotl(s20, 62);                                       \
    s20 = rotl(s22, 43);                                       \
    s22 = rotl(s32, 25);                                       \
    s32 = rotl(s43,  8);                                       \
    s43 = rotl(s34, 56);                                       \
    s34 = rotl(s03, 41);                                       \
    s03 = rotl(s40, 27);                                       \
    s40 = rotl(s44, 14);                                       \
    s44 = rotl(s14,  2);                                       \
    s14 = rotl(s31, 55);                                       \
    s31 = rotl(s13, 45);                                       \
    s13 = rotl(s01, 36);                                       \
    s01 = rotl(s30, 28);                                       \
    s30 = rotl(s33, 21);                                       \
    s33 = rotl(s23, 15);                                       \
    s23 = rotl(s12, 10);                                       \
    s12 = rotl(s21,  6);                                       \
    s21 = rotl(s02,  3);                                       \
    s02 = t0;                                                  \
}

#define KHI(s00, s01, s02, s03, s04,                           \
            s10, s11, s12, s13, s14,                           \
            s20, s21, s22, s23, s24,                           \
            s30, s31, s32, s33, s34,                           \
            s40, s41, s42, s43, s44)                           \
{                                                              \
    t0 = s00; t1 = s10; s00 ^= (~t1) & s20; s10 ^= (~s20) & s30; s20 ^= (~s30) & s40; s30 ^= (~s40) & t0; s40 ^= (~t0) & t1; \
    t0 = s01; t1 = s11; s01 ^= (~t1) & s21; s11 ^= (~s21) & s31; s21 ^= (~s31) & s41; s31 ^= (~s41) & t0; s41 ^= (~t0) & t1; \
    t0 = s02; t1 = s12; s02 ^= (~t1) & s22; s12 ^= (~s22) & s32; s22 ^= (~s32) & s42; s32 ^= (~s42) & t0; s42 ^= (~t0) & t1; \
    t0 = s03; t1 = s13; s03 ^= (~t1) & s23; s13 ^= (~s23) & s33; s23 ^= (~s33) & s43; s33 ^= (~s43) & t0; s43 ^= (~t0) & t1; \
    t0 = s04; t1 = s14; s04 ^= (~t1) & s24; s14 ^= (~s24) & s34; s24 ^= (~s34) & s44; s34 ^= (~s44) & t0; s44 ^= (~t0) & t1; \
}

/* Every round but the last, and the trailing pad byte, which lives here rather
 * than in each ending below so that no ending can be written without it. */
static void keccakf_rounds(thread ulong* st) {
    st[16] ^= 0x8000000000000000UL;
    ulong t0, t1, t2, t3, t4, t5;

    for (int i = 0; i < 23; ++i) {
        THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        st[0] ^= kRoundConstants[i];
    }
}

static void keccakf(thread ulong* st) {
    keccakf_rounds(st);
    ulong t0, t1, t2, t3, t4, t5;

    THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
    RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
    KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
    st[0] ^= kRoundConstants[23];
}

/* Keccak-f for a caller that reads bytes 12 to 32 and nothing else, which is
 * where an Ethereum address comes from. Only lanes 1 to 3 are left correct;
 * every other lane holds a value from the middle of the last round. Anything
 * that reads more of the state than those twenty bytes wants keccakf.
 *
 * The last round is cut to what those three lanes need. Iota only touches lane
 * 0, so it goes entirely. Chi produces lanes 1 to 3 from lanes 0 to 4 alone,
 * and rho/pi builds those five from the diagonal 0, 6, 12, 18, 24, so nineteen
 * of the twenty-four rotations and twenty-two of the twenty-five chi triples
 * are dead. Theta stays whole: those five lanes take all five column parities,
 * and the parities take all 25 lanes.
 *
 * Measured on an M4 Max, macOS 26.5.2, 2026-08-11, alternating order with
 * 60-second cooldowns: create2 730.6 -> 754.1 and create3 357.7 -> 363.1 MH/s,
 * neither pair overlapping. Half a round in twenty-four is worth about 2% and
 * the rest is the last round being peeled out of the loop above, which the
 * profanity kernel gained 1.9% from while still calling the full permutation.
 */
static void keccakf_address(thread ulong* st) {
    keccakf_rounds(st);
    ulong t0, t1, t2, t3, t4, t5;

    THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);

    // The five lanes RHOPI would leave at 0 to 4, then the three of KHI's
    // first group that carry bytes 8 to 32.
    const ulong b0 = st[0];
    const ulong b1 = rotl(st[6], 44);
    const ulong b2 = rotl(st[12], 43);
    const ulong b3 = rotl(st[18], 21);
    const ulong b4 = rotl(st[24], 14);
    st[1] = b1 ^ ((~b2) & b3);
    st[2] = b2 ^ ((~b3) & b4);
    st[3] = b3 ^ ((~b4) & b0);
}

static inline uchar byte_at(thread const ulong* state, uint index) {
    return (uchar)((state[index >> 3] >> ((index & 7) * 8)) & 0xff);
}

static inline void set_byte(thread ulong* state, uint index, uchar value) {
    uint shift = (index & 7) * 8;
    state[index >> 3] &= ~(0xffUL << shift);
    state[index >> 3] |= ((ulong)value) << shift;
}
