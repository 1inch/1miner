/* Metal secp256k1 vanity search, a port of kernels/opencl/profanity.cl.
 *
 * The Keccak-f permutation and the scoring functions come from keccak.metal
 * and scoring.metal, prepended to this source by the host.
 *
 * Cutting corners
 * ===============
 * Carried over from the OpenCL original: the elliptic point addition does not
 * handle two points sharing an X coordinate. It is very unlikely to occur and
 * the performance penalty for doing it right is too severe. A point that hits
 * the case is garbage from then on, costing 1/(i*I) of the hashrate each.
 *
 * Differences from the OpenCL kernel
 * ==================================
 * `profanity_inverse` keeps its prefix products in the output buffer rather
 * than in two private arrays; see the comment on that kernel.
 *
 * There is one scoring entry point reading `Mode.function` at run time instead
 * of eight compiled from a macro. OpenCL specialises at build time because it
 * gets a `-D` for free; a Metal library is built from source at run time and a
 * second pipeline would be a second compile.
 *
 * Every work item id is `thread_position_in_grid` plus `params.idBase`. Metal
 * has no equivalent of `set_global_work_offset`, so a launch split into chunks
 * would otherwise start each chunk's ids at zero, and the offset reported for
 * a hit names the private key the user ends up with.
 */

#define MP_WORDS 8

// Eight 32-bit words, least significant first, exactly as profanity2's
// mp_number. Thirty-two bytes, so the array stride matches the host's
// #[repr(C, align(16))] whatever alignment the compiler picks.
struct MpNumber {
    uint d[MP_WORDS];
};

struct MpPoint {
    MpNumber x;
    MpNumber y;
};

struct ProfResult {
    uint found;
    uint foundId;
    uchar foundHash[20];
};

struct ProfParams {
    ulong seed[4];   // this device's starting scalar, least significant lane first
    ulong seedX[4];  // the user's seed public key
    ulong seedY[4];
    uint idBase;     // added to thread_position_in_grid; see the file header
    uint inverseSize;
    uint scoreMax;
    uint patternCount;   // --exact only
    uint exactCapacity;  // --exact only: highest slot the exact kernel fills
    uint contract;       // score the contract this key deploys at nonce 0
};

// mod              = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
constant MpNumber kMod              = { {0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

// tripleNegativeGx = 0x92c4cc831269ccfaff1ed83e946adeeaf82c096e76958573f2287becbb17b196
constant MpNumber kTripleNegativeGx = { {0xbb17b196, 0xf2287bec, 0x76958573, 0xf82c096e, 0x946adeea, 0xff1ed83e, 0x1269ccfa, 0x92c4cc83} };

// negativeGy       = 0xb7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777
constant MpNumber kNegativeGy       = { {0x04ef2777, 0x63b82f6f, 0x597aabe6, 0x02e84bb7, 0xf1eef757, 0xa25b0403, 0xd95c3b9a, 0xb7c52588} };

#define bswap32(n) (rotate((n) & 0x00FF00FFu, 24u) | (rotate((n), 8u) & 0x00FF00FFu))

/* ------------------------------------------------------------------------ */
/* Device memory moves                                                      */
/* ------------------------------------------------------------------------ */
/* Written out word by word rather than as struct assignment so that the
 * address space of each side is stated at the point of the move. */

static inline void load_mp(thread MpNumber* dst, device const MpNumber* src) {
    for (uint i = 0; i < MP_WORDS; ++i) {
        dst->d[i] = src->d[i];
    }
}

static inline void store_mp(device MpNumber* dst, thread const MpNumber* src) {
    for (uint i = 0; i < MP_WORDS; ++i) {
        dst->d[i] = src->d[i];
    }
}

static inline void load_point(thread MpPoint* dst, device const MpPoint* src) {
    load_mp(&dst->x, &src->x);
    load_mp(&dst->y, &src->y);
}

/* ------------------------------------------------------------------------ */
/* Multiprecision functions                                                 */
/* ------------------------------------------------------------------------ */

// Multiprecision subtraction. Underflow signalled via return value.
static uint mp_sub(thread MpNumber* r, thread const MpNumber* a, thread const MpNumber* b) {
    uint t, c = 0;

    for (uint i = 0; i < MP_WORDS; ++i) {
        t = a->d[i] - b->d[i] - c;
        c = t > a->d[i] ? 1 : (t == a->d[i] ? c : 0);

        r->d[i] = t;
    }

    return c;
}

// Multiprecision subtraction of the modulus. Underflow signalled via return value.
static uint mp_sub_mod(thread MpNumber* r) {
    uint t, c = 0;

    for (uint i = 0; i < MP_WORDS; ++i) {
        t = r->d[i] - kMod.d[i] - c;
        c = t > r->d[i] ? 1 : (t == r->d[i] ? c : 0);

        r->d[i] = t;
    }

    return c;
}

// Multiprecision subtraction modulo M, M = kMod.
// This function is often also used for additions by subtracting a negative
// number, because a modular addition would have to determine whether the result
// is in the gap M <= r < 2^256 while a subtraction signals underflow in its
// carry for free.
static void mp_mod_sub(thread MpNumber* r, thread const MpNumber* a, thread const MpNumber* b) {
    uint i, t, c = 0;

    for (i = 0; i < MP_WORDS; ++i) {
        t = a->d[i] - b->d[i] - c;
        c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

        r->d[i] = t;
    }

    if (c) {
        c = 0;
        for (i = 0; i < MP_WORDS; ++i) {
            r->d[i] += kMod.d[i] + c;
            c = r->d[i] < kMod.d[i] ? 1 : (r->d[i] == kMod.d[i] ? c : 0);
        }
    }
}

// Multiprecision subtraction modulo M from a number in the constant address space.
static void mp_mod_sub_const(thread MpNumber* r, constant const MpNumber* a, thread const MpNumber* b) {
    uint i, t, c = 0;

    for (i = 0; i < MP_WORDS; ++i) {
        t = a->d[i] - b->d[i] - c;
        c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

        r->d[i] = t;
    }

    if (c) {
        c = 0;
        for (i = 0; i < MP_WORDS; ++i) {
            r->d[i] += kMod.d[i] + c;
            c = r->d[i] < kMod.d[i] ? 1 : (r->d[i] == kMod.d[i] ? c : 0);
        }
    }
}

// Multiprecision subtraction modulo M of G_x from a number.
// Specialization of mp_mod_sub in hope of performance gain.
static void mp_mod_sub_gx(thread MpNumber* r, thread const MpNumber* a) {
    uint i, t, c = 0;

    t = a->d[0] - 0x16f81798; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
    t = a->d[1] - 0x59f2815b - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
    t = a->d[2] - 0x2dce28d9 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
    t = a->d[3] - 0x029bfcdb - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
    t = a->d[4] - 0xce870b07 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
    t = a->d[5] - 0x55a06295 - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
    t = a->d[6] - 0xf9dcbbac - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
    t = a->d[7] - 0x79be667e - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

    if (c) {
        c = 0;
        for (i = 0; i < MP_WORDS; ++i) {
            r->d[i] += kMod.d[i] + c;
            c = r->d[i] < kMod.d[i] ? 1 : (r->d[i] == kMod.d[i] ? c : 0);
        }
    }
}

// Multiprecision subtraction modulo M of G_y from a number.
// Specialization of mp_mod_sub in hope of performance gain.
static void mp_mod_sub_gy(thread MpNumber* r, thread const MpNumber* a) {
    uint i, t, c = 0;

    t = a->d[0] - 0xfb10d4b8; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
    t = a->d[1] - 0x9c47d08f - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
    t = a->d[2] - 0xa6855419 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
    t = a->d[3] - 0xfd17b448 - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
    t = a->d[4] - 0x0e1108a8 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
    t = a->d[5] - 0x5da4fbfc - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
    t = a->d[6] - 0x26a3c465 - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
    t = a->d[7] - 0x483ada77 - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

    if (c) {
        c = 0;
        for (i = 0; i < MP_WORDS; ++i) {
            r->d[i] += kMod.d[i] + c;
            c = r->d[i] < kMod.d[i] ? 1 : (r->d[i] == kMod.d[i] ? c : 0);
        }
    }
}

// Multiprecision addition. Overflow signalled via return value.
static uint mp_add(thread MpNumber* r, thread const MpNumber* a) {
    uint c = 0;

    for (uint i = 0; i < MP_WORDS; ++i) {
        r->d[i] += a->d[i] + c;
        c = r->d[i] < a->d[i] ? 1 : (r->d[i] == a->d[i] ? c : 0);
    }

    return c;
}

// Multiprecision addition of the modulus. Overflow signalled via return value.
static uint mp_add_mod(thread MpNumber* r) {
    uint c = 0;

    for (uint i = 0; i < MP_WORDS; ++i) {
        r->d[i] += kMod.d[i] + c;
        c = r->d[i] < kMod.d[i] ? 1 : (r->d[i] == kMod.d[i] ? c : 0);
    }

    return c;
}

// Multiprecision addition of two numbers with one extra word each. Overflow signalled via return value.
static uint mp_add_more(thread MpNumber* r, thread uint* extraR, thread const MpNumber* a, thread const uint* extraA) {
    const uint c = mp_add(r, a);
    *extraR += *extraA + c;
    return *extraR < *extraA ? 1 : (*extraR == *extraA ? c : 0);
}

// Multiprecision greater than or equal (>=) operator
static uint mp_gte(thread const MpNumber* a, thread const MpNumber* b) {
    uint l = 0, g = 0;

    for (uint i = 0; i < MP_WORDS; ++i) {
        if (a->d[i] < b->d[i]) l |= (1 << i);
        if (a->d[i] > b->d[i]) g |= (1 << i);
    }

    return g >= l;
}

// Bit shifts a number with an extra word to the right one step
static void mp_shr_extra(thread MpNumber* r, thread uint* e) {
    r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
    r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
    r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
    r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
    r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
    r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
    r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
    r->d[7] = (*e << 31) | (r->d[7] >> 1);
    *e >>= 1;
}

// Bit shifts a number to the right one step
static void mp_shr(thread MpNumber* r) {
    r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
    r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
    r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
    r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
    r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
    r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
    r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
    r->d[7] >>= 1;
}

// Multiplies a number with a word and adds it to an existing number with an extra word,
// overflow of the extra word is signalled in the return value.
// This is a special function only used for modular multiplication.
static uint mp_mul_word_add_extra(thread MpNumber* r, thread const MpNumber* a, const uint w, thread uint* extra) {
    uint cM = 0; // Carry for multiplication
    uint cA = 0; // Carry for addition
    uint tM = 0; // Temporary storage for multiplication

    for (uint i = 0; i < MP_WORDS; ++i) {
        tM = (a->d[i] * w + cM);
        cM = mulhi(a->d[i], w) + (tM < cM);

        r->d[i] += tM + cA;
        cA = r->d[i] < tM ? 1 : (r->d[i] == tM ? cA : 0);
    }

    *extra += cM + cA;
    return *extra < cM ? 1 : (*extra == cM ? cA : 0);
}

// Multiplies a number with a word, potentially adds modhigher to it, and then subtracts it from
// an existing number, no extra words, no overflow.
//
// This is a special function only used for modular multiplication.
//
// Optimization (secp256k1 fast reduction) contributed by Rodrigo Madera (@madera).
//
// The secp256k1 prime has the special form:
//
//   p = 2^256 - pmod, where pmod = 2^32 + 977 = 0x1000003D1
//
// Therefore, working modulo 2^256:
//
//   q * p = q * (2^256 - pmod) = q * 2^256 - q * pmod == -q * pmod  (mod 2^256)
//
//   (r - q * p) mod 2^256 == (r + q * pmod) mod 2^256
//
// So instead of multiplying q by the full 256-bit p and subtracting, we multiply q by the
// 33-bit pmod and add. This reduces the amount of bits used, giving us 20-35% speed improvements.
//
static void mp_mul_mod_word_sub(thread MpNumber* r, const uint w, const bool withModHigher) {
    const uint lo977 = 977u * w;
    const uint hi977 = mulhi(977u, w);

    const uint p0 = lo977;
    const ulong p1_full = (ulong)w + hi977 + (withModHigher ? 0x000003D1u : 0u);
    const uint p1 = (uint)p1_full;
    const uint p2 = (uint)(p1_full >> 32) + (withModHigher ? 1u : 0u);

    ulong s = (ulong)r->d[0] + p0;
    r->d[0] = (uint)s;
    uint c = (uint)(s >> 32);

    s = (ulong)r->d[1] + p1 + c;
    r->d[1] = (uint)s;
    c = (uint)(s >> 32);

    s = (ulong)r->d[2] + p2 + c;
    r->d[2] = (uint)s;
    c = (uint)(s >> 32);

    for (uint i = 3; i < MP_WORDS; ++i) {
        s = (ulong)r->d[i] + c;
        r->d[i] = (uint)s;
        c = (uint)(s >> 32);
    }
}

// Modular multiplication. Based on Algorithm 3 from this article:
// https://www.esat.kuleuven.be/cosic/publications/article-1191.pdf
// The additional end steps of adding or subtracting the modulo never came up in
// the original author's testing; see "Cutting corners" at the top of this file.
static void mp_mod_mul(thread MpNumber* r, thread const MpNumber* X, thread const MpNumber* Y) {
    MpNumber Z = { {0} };
    uint extraWord;

    for (int i = MP_WORDS - 1; i >= 0; --i) {
        // Z = Z * 2^32
        extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;

        // Z = Z + X * Y_i
        bool overflow = mp_mul_word_add_extra(&Z, X, Y->d[i], &extraWord);

        // Z = Z - qM
        mp_mul_mod_word_sub(&Z, extraWord, overflow);
    }

    *r = Z;
}

// Modular inversion of a number.
static void mp_mod_inverse(thread MpNumber* r) {
    MpNumber A = { { 1 } };
    MpNumber C = { { 0 } };
    MpNumber v;
    for (uint i = 0; i < MP_WORDS; ++i) {
        v.d[i] = kMod.d[i];
    }

    uint extraA = 0;
    uint extraC = 0;

    while (r->d[0] || r->d[1] || r->d[2] || r->d[3] || r->d[4] || r->d[5] || r->d[6] || r->d[7]) {
        while (!(r->d[0] & 1)) {
            mp_shr(r);
            if (A.d[0] & 1) {
                extraA += mp_add_mod(&A);
            }

            mp_shr_extra(&A, &extraA);
        }

        while (!(v.d[0] & 1)) {
            mp_shr(&v);
            if (C.d[0] & 1) {
                extraC += mp_add_mod(&C);
            }

            mp_shr_extra(&C, &extraC);
        }

        if (mp_gte(r, &v)) {
            mp_sub(r, r, &v);
            mp_add_more(&A, &extraA, &C, &extraC);
        }
        else {
            mp_sub(&v, &v, r);
            mp_add_more(&C, &extraC, &A, &extraA);
        }
    }

    while (extraC) {
        extraC -= mp_sub_mod(&C);
    }

    for (uint i = 0; i < MP_WORDS; ++i) {
        v.d[i] = kMod.d[i];
    }
    mp_sub(r, &v, &C);
}

/* ------------------------------------------------------------------------ */
/* Elliptic point and addition (with caveats).                              */
/* ------------------------------------------------------------------------ */

// Elliptical point addition
// Does not handle points sharing X coordinate, this is a deliberate design choice.
// For more information on this choice see the beginning of this file.
static void point_add(thread MpPoint* r, thread MpPoint* p, thread MpPoint* o) {
    MpNumber tmp;
    MpNumber newX;
    MpNumber newY;

    mp_mod_sub(&tmp, &o->x, &p->x);

    mp_mod_inverse(&tmp);

    mp_mod_sub(&newX, &o->y, &p->y);
    mp_mod_mul(&tmp, &tmp, &newX);

    mp_mod_mul(&newX, &tmp, &tmp);
    mp_mod_sub(&newX, &newX, &p->x);
    mp_mod_sub(&newX, &newX, &o->x);

    mp_mod_sub(&newY, &p->x, &newX);
    mp_mod_mul(&newY, &newY, &tmp);
    mp_mod_sub(&newY, &newY, &p->y);

    r->x = newX;
    r->y = newY;
}

/* ------------------------------------------------------------------------ */
/* Profanity.                                                               */
/* ------------------------------------------------------------------------ */

static void profanity_init_seed(
    device const MpPoint* precomp,
    thread MpPoint* p,
    thread bool* pIsFirst,
    const uint precompOffset,
    const ulong seed)
{
    MpPoint o;

    for (uint i = 0; i < 8; ++i) {
        const uint shift = i * 8;
        const uint byte = (uint)((seed >> shift) & 0xFF);

        if (byte) {
            load_point(&o, &precomp[precompOffset + i * 255 + byte - 1]);
            if (*pIsFirst) {
                *p = o;
                *pIsFirst = false;
            }
            else {
                point_add(p, p, &o);
            }
        }
    }
}

/// Unpack four 64-bit lanes, least significant first, into eight 32-bit words.
static void lanes_to_mp(thread MpNumber* r, constant const ulong* lanes) {
    for (uint i = 0; i < 4; ++i) {
        r->d[i * 2] = (uint)(lanes[i] & 0xFFFFFFFF);
        r->d[i * 2 + 1] = (uint)(lanes[i] >> 32);
    }
}

kernel void profanity_init(
    device const MpPoint* precomp [[buffer(0)]],
    device MpNumber* pDeltaX      [[buffer(1)]],
    device MpNumber* pPrevLambda  [[buffer(2)]],
    constant ProfParams& params   [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]])
{
    const uint id = gid + params.idBase;

    MpPoint p;
    lanes_to_mp(&p.x, params.seedX);
    lanes_to_mp(&p.y, params.seedY);

    // Zeroed rather than left undefined as in the OpenCL original: a seed whose
    // every byte is zero would leave this unread, and 192 random bits make that
    // a case that never arrives rather than one worth branching on.
    MpPoint pRandom;
    for (uint i = 0; i < MP_WORDS; ++i) {
        pRandom.x.d[i] = 0;
        pRandom.y.d[i] = 0;
    }
    bool bIsFirst = true;

    MpNumber tmp1, tmp2;
    MpPoint tmp3;

    // Calculate k*G where k is the scalar in params.seed, in other words find
    // the point indicated by the private key that scalar represents.
    profanity_init_seed(precomp, &pRandom, &bIsFirst, 8 * 255 * 0, params.seed[0]);
    profanity_init_seed(precomp, &pRandom, &bIsFirst, 8 * 255 * 1, params.seed[1]);
    profanity_init_seed(precomp, &pRandom, &bIsFirst, 8 * 255 * 2, params.seed[2]);
    profanity_init_seed(precomp, &pRandom, &bIsFirst, 8 * 255 * 3, params.seed[3] + (ulong)id);
    point_add(&p, &p, &pRandom);

    // Calculate current lambda in this point
    mp_mod_sub_gx(&tmp1, &p.x);
    mp_mod_inverse(&tmp1);

    mp_mod_sub_gy(&tmp2, &p.y);
    mp_mod_mul(&tmp1, &tmp1, &tmp2);

    // Jump to next point (precomp[0] is the generator point G)
    load_point(&tmp3, &precomp[0]);
    point_add(&p, &tmp3, &p);

    // pDeltaX should contain the delta (x - G_x)
    mp_mod_sub_gx(&p.x, &p.x);

    store_mp(&pDeltaX[id], &p.x);
    store_mp(&pPrevLambda[id], &tmp1);
}

/* This kernel calculates several modular inversions at once with just one
 * inverse. It's an implementation of Algorithm 2.11 from Modern Computer
 * Arithmetic: https://members.loria.fr/PZimmermann/mca/pub226.html
 *
 * The OpenCL kernel holds the prefix products in two private mp_number arrays
 * of PROFANITY_INVERSE_SIZE entries, which at the default width is 16 KB of
 * private memory per work item. An Apple GPU spills that, so here the prefix
 * products go into pInverse, which is written anyway, and the second array is
 * replaced by re-reading pDeltaX. Each work item owns a contiguous range, and
 * the backward pass reads prefix i-1 before overwriting slot i, so nothing is
 * read after it has been clobbered. The arithmetic is unchanged, which is what
 * the cross-backend agreement tests check.
 *
 * inverseSize arrives in a buffer rather than as a -D because there is no
 * longer an array to size with it.
 */
kernel void profanity_inverse(
    device const MpNumber* pDeltaX [[buffer(0)]],
    device MpNumber* pInverse      [[buffer(1)]],
    constant ProfParams& params    [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]])
{
    // negativeDoubleGy = 0x6f8a4b11b2b8773544b60807e3ddeeae05d0976eb2f557ccc7705edf09de52bf
    MpNumber negativeDoubleGy = { {0x09de52bf, 0xc7705edf, 0xb2f557cc, 0x05d0976e, 0xe3ddeeae, 0x44b60807, 0xb2b87735, 0x6f8a4b11} };

    const uint size = params.inverseSize;
    const uint base = (gid + params.idBase) * size;

    MpNumber acc, cur, prefix, out;

    // pInverse[base + i] = pDeltaX[base] * pDeltaX[base + 1] * ... * pDeltaX[base + i]
    load_mp(&acc, &pDeltaX[base]);
    store_mp(&pInverse[base], &acc);
    for (uint i = 1; i < size; ++i) {
        load_mp(&cur, &pDeltaX[base + i]);
        mp_mod_mul(&acc, &cur, &acc);
        store_mp(&pInverse[base + i], &acc);
    }

    // Take the inverse of all x-values combined, then multiply in -2G_y so that
    // we have -2G_y / (x_0 * x_1 * x_2 * ...). Folding the constant in here
    // costs one multiplication per batch instead of one per point.
    mp_mod_inverse(&acc);
    mp_mod_mul(&acc, &acc, &negativeDoubleGy);

    // Multiply out each individual inverse, consuming the prefixes as we go.
    for (uint i = size - 1; i > 0; --i) {
        load_mp(&prefix, &pInverse[base + i - 1]);
        mp_mod_mul(&out, &acc, &prefix);
        load_mp(&cur, &pDeltaX[base + i]);
        mp_mod_mul(&acc, &acc, &cur);
        store_mp(&pInverse[base + i], &out);
    }

    store_mp(&pInverse[base], &acc);
}

/* This performs an elliptical curve point addition. See:
 * https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_addition
 *
 * The original author made one mathematical optimization by never calculating
 * x_r, instead directly calculating the delta (x_q - x_p). It's for this delta
 * that the inverse is calculated, and that has already been done by the kernel
 * above. Storing the next delta saves one mp_mod_sub per point, at the cost of
 * an addition below to retrieve the actual x-coordinate for hashing.
 *
 * With d the delta and d' the new delta, x_p as G_x and y_p as G_y:
 *
 *   d = x - G_x <=> x = d + G_x
 *   x' = l² - G_x - x = l² - 2G_x - d
 *   d' = x' - G_x = l² - 3G_x - d
 *
 * so d' costs the same as x' would, 3G_x being just another constant. For the
 * next y-coordinate, y' = l(G_x - x') - G_y, and G_x - x' = -(x' - G_x) = -d',
 * so y' = -G_y - (l * d') removes another subtraction. Then:
 *
 *   l' = (y' - G_y) / d' = (-l * d' - 2G_y) / d' = -l - 2G_y / d'
 *
 * The -2G_y term is constant, which is why it can be multiplied in during the
 * inversion once per batch. Point addition is thus down to two multiprecision
 * multiplications from three. The y-coordinate is never stored, only
 * recalculated at the end from the lambda and the delta.
 *
 * After the point addition this calculates the public address corresponding to
 * the point and returns it in private memory, where the scoring grades it
 * without a round trip through device memory.
 */
static void profanity_iterate(
    device MpNumber* pDeltaX,
    device const MpNumber* pInverse,
    device MpNumber* pPrevLambda,
    const uint id,
    const uint contract,
    thread uchar* address)
{
    // negativeGx = 0x8641998106234453aa5f9d6a3178f4f8fd640324d231d726a60d7ea3e907e497
    MpNumber negativeGx = { {0xe907e497, 0xa60d7ea3, 0xd231d726, 0xfd640324, 0x3178f4f8, 0xaa5f9d6a, 0x06234453, 0x86419981} };

    MpNumber dX, tmp, lambda;
    load_mp(&dX, &pDeltaX[id]);
    load_mp(&tmp, &pInverse[id]);
    load_mp(&lambda, &pPrevLambda[id]);

    // l' = -(2G_y) / d' - l
    mp_mod_sub(&lambda, &tmp, &lambda);

    // l² = l * l
    mp_mod_mul(&tmp, &lambda, &lambda);

    // d' = l² - d - 3g = (-3g) - (d - l²)
    mp_mod_sub(&dX, &dX, &tmp);
    mp_mod_sub_const(&dX, &kTripleNegativeGx, &dX);

    store_mp(&pDeltaX[id], &dX);
    store_mp(&pPrevLambda[id], &lambda);

    // Calculate y from dX and lambda: y' = (-G_y) - l * d'
    mp_mod_mul(&tmp, &lambda, &dX);
    mp_mod_sub_const(&tmp, &kNegativeGy, &tmp);

    // Restore X coordinate from delta value
    mp_mod_sub(&dX, &dX, &negativeGx);

    // Initialize the keccak state with the point coordinates in big endian.
    // The OpenCL kernel writes 32-bit words into an ethhash union; the same
    // little-endian aliasing is spelled out here by packing word pairs.
    uint words[16];
    for (uint i = 0; i < 8; ++i) {
        words[i] = bswap32(dX.d[MP_WORDS - 1 - i]);
        words[8 + i] = bswap32(tmp.d[MP_WORDS - 1 - i]);
    }

    ulong st[25];
    for (uint i = 0; i < 8; ++i) {
        st[i] = ((ulong)words[i * 2 + 1] << 32) | (ulong)words[i * 2];
    }
    for (uint i = 8; i < 25; ++i) {
        st[i] = 0;
    }
    st[8] ^= 0x01; // length 64: the leading keccak pad bit

    // keccakf_address would be valid at both hashes here, for the reason and
    // with the measurement kernels/opencl/profanity.cl records: 269.9 against
    // 267.1 MH/s under --contract on an M4 Max, macOS 26.5.2, 2026-08-11, both
    // pairs the wrong way round.
    keccakf(st);

    // The address is the low 20 bytes of the hash.
    for (uint i = 0; i < 20; ++i) {
        address[i] = byte_at(st, i + 12);
    }

    if (contract != 0) {
        // keccak(0xd6, 0x94, address, 0x80): the contract this key deploys at
        // nonce 0, where 0x80 is the RLP encoding of that nonce.
        ulong second[25];
        for (uint i = 0; i < 25; ++i) {
            second[i] = 0;
        }
        set_byte(second, 0, 0xd6);
        set_byte(second, 1, 0x94);
        for (uint i = 0; i < 20; ++i) {
            set_byte(second, 2 + i, address[i]);
        }
        set_byte(second, 22, 0x80);
        set_byte(second, 23, 0x01); // length 23: the leading keccak pad bit
        keccakf(second);

        for (uint i = 0; i < 20; ++i) {
            address[i] = byte_at(second, i + 12);
        }
    }
}

kernel void profanity_iterate_score(
    device MpNumber* pDeltaX       [[buffer(0)]],
    device const MpNumber* pInverse [[buffer(1)]],
    device MpNumber* pPrevLambda   [[buffer(2)]],
    device ProfResult* results     [[buffer(3)]],
    constant Mode& mode            [[buffer(4)]],
    constant ProfParams& params    [[buffer(5)]],
    device atomic_uint* foundFlags [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]])
{
    const uint id = gid + params.idBase;
    uchar address[20];
    profanity_iterate(pDeltaX, pInverse, pPrevLambda, id, params.contract, address);

    int score = score_address(address, mode);
    if (score <= 0 || (uint)score <= params.scoreMax) {
        return;
    }

    // One slot per score, first writer wins.
    if (atomic_fetch_add_explicit(&foundFlags[score], 1u, memory_order_relaxed) != 0) {
        return;
    }
    results[score].foundId = id;
    for (uint i = 0; i < 20; ++i) {
        results[score].foundHash[i] = address[i];
    }
    results[score].found = 1;
}

/// --exact: every address matching one of the masks in full, appended in
/// arrival order rather than kept one per score.
///
/// A different question from profanity_iterate_score and so a different buffer.
/// There is no score and no bar: a candidate either matches completely or is
/// not a result, and all of them are wanted, which one slot per score cannot
/// express.
///
/// foundFlags[0] counts this round's matches and results[1..exactCapacity] hold
/// them. The host clears the counter before each round, so the bounds check is
/// also what keeps two threads out of one slot. Matches past the capacity are
/// counted but not stored, and the host reports how many.
kernel void profanity_iterate_exact_match(
    device MpNumber* pDeltaX       [[buffer(0)]],
    device const MpNumber* pInverse [[buffer(1)]],
    device MpNumber* pPrevLambda   [[buffer(2)]],
    device ProfResult* results     [[buffer(3)]],
    constant Pattern* patterns     [[buffer(4)]],
    constant ProfParams& params    [[buffer(5)]],
    device atomic_uint* foundFlags [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]])
{
    const uint id = gid + params.idBase;
    uchar address[20];
    profanity_iterate(pDeltaX, pInverse, pPrevLambda, id, params.contract, address);

    for (uint p = 0; p < params.patternCount; ++p) {
        if (!matches_pattern(address, patterns[p])) {
            continue;
        }

        uint slot = atomic_fetch_add_explicit(&foundFlags[0], 1u, memory_order_relaxed) + 1;
        if (slot <= params.exactCapacity) {
            results[slot].foundId = id;
            for (uint i = 0; i < 20; ++i) {
                results[slot].foundHash[i] = address[i];
            }
            results[slot].found = p + 1;
        }
        // One address can satisfy two masks; report it once, against the first.
        return;
    }
}
