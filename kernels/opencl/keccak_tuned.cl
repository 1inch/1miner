/* This Keccak implementation is an amalgamation of:
 * Tiny SHA3 implementation by Markku-Juhani O. Saarinen:
 *   https://github.com/mjosaarinen/tiny_sha3
 * Keccak implementation found in xptMiner-gpu @ Github:
 * https://github.com/llamasoft/xptMiner-gpu/blob/master/opencl/keccak.cl
 */

typedef union {
	uchar b[200];
	ulong q[25];
	uint d[50];
} ethhash;

#define TH_ELT_SHORT(t, d, c) t = rotate(d, (ulong) 1) ^ c

#define THETA(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
	t0 = s00 ^ s01 ^ s02 ^ s03 ^ s04;                      \
	t1 = s10 ^ s11 ^ s12 ^ s13 ^ s14;                      \
	t2 = s20 ^ s21 ^ s22 ^ s23 ^ s24;                      \
	t3 = s30 ^ s31 ^ s32 ^ s33 ^ s34;                      \
	t4 = s40 ^ s41 ^ s42 ^ s43 ^ s44;                      \
                                                           \
	TH_ELT_SHORT(t5, t0, t3);                              \
	TH_ELT_SHORT(t0, t2, t0);                              \
	TH_ELT_SHORT(t2, t4, t2);                              \
	TH_ELT_SHORT(t4, t1, t4);                              \
	TH_ELT_SHORT(t1, t3, t1);                              \
                                                           \
    s00 ^= t4; s01 ^= t4; s02 ^= t4; s03 ^= t4; s04 ^= t4; \
    s10 ^= t0; s11 ^= t0; s12 ^= t0; s13 ^= t0; s14 ^= t0; \
    s20 ^= t1; s21 ^= t1; s22 ^= t1; s23 ^= t1; s24 ^= t1; \
    s30 ^= t2; s31 ^= t2; s32 ^= t2; s33 ^= t2; s34 ^= t2; \
    s40 ^= t5; s41 ^= t5; s42 ^= t5; s43 ^= t5; s44 ^= t5; \
}

#define RHOPI(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
	t0  = rotate(s10, (ulong) 1);  \
	s10 = rotate(s11, (ulong)44); \
	s11 = rotate(s41, (ulong)20); \
	s41 = rotate(s24, (ulong)61); \
	s24 = rotate(s42, (ulong)39); \
	s42 = rotate(s04, (ulong)18); \
	s04 = rotate(s20, (ulong)62); \
	s20 = rotate(s22, (ulong)43); \
	s22 = rotate(s32, (ulong)25); \
	s32 = rotate(s43, (ulong) 8); \
	s43 = rotate(s34, (ulong)56); \
	s34 = rotate(s03, (ulong)41); \
	s03 = rotate(s40, (ulong)27); \
	s40 = rotate(s44, (ulong)14); \
	s44 = rotate(s14, (ulong) 2); \
	s14 = rotate(s31, (ulong)55); \
	s31 = rotate(s13, (ulong)45); \
	s13 = rotate(s01, (ulong)36); \
	s01 = rotate(s30, (ulong)28); \
	s30 = rotate(s33, (ulong)21); \
	s33 = rotate(s23, (ulong)15); \
	s23 = rotate(s12, (ulong)10); \
	s12 = rotate(s21, (ulong) 6); \
	s21 = rotate(s02, (ulong) 3); \
	s02 = t0; \
}

#define KHI(s00, s01, s02, s03, s04, \
            s10, s11, s12, s13, s14, \
            s20, s21, s22, s23, s24, \
            s30, s31, s32, s33, s34, \
            s40, s41, s42, s43, s44) \
{ \
	t0 = s00; t1 = s10; s00 ^= (~t1) & s20; s10 ^= (~s20) & s30; s20 ^= (~s30) & s40; s30 ^= (~s40) & t0; s40 ^= (~t0) & t1; \
	t0 = s01; t1 = s11; s01 ^= (~t1) & s21; s11 ^= (~s21) & s31; s21 ^= (~s31) & s41; s31 ^= (~s41) & t0; s41 ^= (~t0) & t1; \
	t0 = s02; t1 = s12; s02 ^= (~t1) & s22; s12 ^= (~s22) & s32; s22 ^= (~s32) & s42; s32 ^= (~s42) & t0; s42 ^= (~t0) & t1; \
	t0 = s03; t1 = s13; s03 ^= (~t1) & s23; s13 ^= (~s23) & s33; s23 ^= (~s33) & s43; s33 ^= (~s43) & t0; s43 ^= (~t0) & t1; \
	t0 = s04; t1 = s14; s04 ^= (~t1) & s24; s14 ^= (~s24) & s34; s24 ^= (~s34) & s44; s34 ^= (~s44) & t0; s44 ^= (~t0) & t1; \
}

#define IOTA(s00, r) { s00 ^= r; }

__constant ulong keccakf_rndc[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// Barely a bottleneck. No need to tinker more.
//
// Every round but the last, and the trailing pad byte, which lives here rather
// than in each ending below so that no ending can be written without it.
void sha3_keccakf_rounds(ethhash * const h)
{
	ulong * const st = h->q;
	h->d[33] ^= 0x80000000;
	ulong t0, t1, t2, t3, t4, t5;

	// Unrolling and removing PI stage gave negligable performance on GTX 1070.
	for (int i = 0; i < 23; ++i) {
		THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		IOTA(st[0], keccakf_rndc[i]);
	}
}

void sha3_keccakf(ethhash * const h)
{
	sha3_keccakf_rounds(h);

	ulong * const st = h->q;
	ulong t0, t1, t2, t3, t4, t5;

	THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	IOTA(st[0], keccakf_rndc[23]);
}

/* Keccak-f for a caller that reads bytes 12 to 32 and nothing else, which is
 * where an Ethereum address comes from. Only lanes 1 to 3 are left correct;
 * every other lane holds a value from the middle of the last round. Anything
 * that reads more of the state than those twenty bytes wants sha3_keccakf.
 *
 * The last round is cut to what those three lanes need. Iota only touches lane
 * 0, so it goes entirely. Chi produces lanes 1 to 3 from lanes 0 to 4 alone,
 * and rho/pi builds those five from the diagonal 0, 6, 12, 18, 24, so
 * nineteen of the twenty-four rotations and twenty-two of the twenty-five chi
 * triples are dead. Theta stays whole: those five lanes take all five column
 * parities, and the parities take all 25 lanes.
 *
 * Measured on an M4 Max, macOS 26.5.2, 2026-08-11, alternating order with
 * 60-second cooldowns: create3 361.8 -> 371.5 and create2 728.7 -> 742.7 MH/s,
 * neither pair overlapping. Half a round in twenty-four is worth about 2% and
 * the rest is the last round being peeled out of the loop above, which the
 * profanity kernel gained 1.2% from while still calling the full permutation.
 */
void sha3_keccakf_address(ethhash * const h)
{
	sha3_keccakf_rounds(h);

	ulong * const st = h->q;
	ulong t0, t1, t2, t3, t4, t5;

	THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);

	// The five lanes RHOPI would leave at 0 to 4, then the three of KHI's
	// first group that carry bytes 8 to 32.
	const ulong b0 = st[0];
	const ulong b1 = rotate(st[6], (ulong)44);
	const ulong b2 = rotate(st[12], (ulong)43);
	const ulong b3 = rotate(st[18], (ulong)21);
	const ulong b4 = rotate(st[24], (ulong)14);
	st[1] = b1 ^ ((~b2) & b3);
	st[2] = b2 ^ ((~b3) & b4);
	st[3] = b3 ^ ((~b4) & b0);
}
