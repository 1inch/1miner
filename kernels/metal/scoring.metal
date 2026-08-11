/* Scoring for the Metal backend, shared by the salt and profanity kernels.
 *
 * OpenCL carries two copies of these functions, one in salt.cl and one in
 * profanity.cl, because each compiles a separate program. Metal builds one
 * library per run out of concatenated sources, so both kernels can read the
 * same scorer and neither can drift from the other.
 *
 * Prepended after keccak.metal, which supplies the `#include`.
 */

struct Mode {
    uint function;
    uchar data1[20];
    uchar data2[20];
};

/// One --exact mask: `mask` has 0xF nibbles where a digit was given and 0 where
/// it was a wildcard, `want` the digits, so a candidate matches when
/// (address[i] & mask[i]) == want[i] across all twenty bytes.
struct Pattern {
    uchar mask[20];
    uchar want[20];
};

// Scoring functions, matching the ScoreFn enum shared with OpenCL.
constant uint kBenchmark = 0;
constant uint kZeroBytes = 1;
constant uint kMatching = 2;
constant uint kLeading = 3;
constant uint kRange = 4;
constant uint kMirror = 5;
constant uint kDoubles = 6;
constant uint kLeadingRange = 7;

static int score_address(thread const uchar* hash, constant Mode& mode) {
    int score = 0;
    switch (mode.function) {
    // A constant 0 is safe because this switch reads mode.function at run time,
    // so every branch survives and the hash stays live. profanity.cl's
    // benchmark scorer has to consume the address bytes instead: it is selected
    // at compile time, where a constant would let its keccak be eliminated and
    // the reported hashrate become fiction. The asymmetry is deliberate.
    case kBenchmark:
        return 0;

    case kZeroBytes:
        for (int i = 0; i < 20; ++i) {
            score += (hash[i] == 0) ? 1 : 0;
        }
        return score;

    case kMatching:
        for (int i = 0; i < 20; ++i) {
            if (mode.data1[i] > 0 && (hash[i] & mode.data1[i]) == mode.data2[i]) {
                ++score;
            }
        }
        return score;

    case kLeading:
        for (int i = 0; i < 20; ++i) {
            if (((hash[i] & 0xF0) >> 4) != mode.data1[0]) { return score; }
            ++score;
            if ((hash[i] & 0x0F) != mode.data1[0]) { return score; }
            ++score;
        }
        return score;

    case kRange:
        for (int i = 0; i < 20; ++i) {
            uchar hi = (hash[i] & 0xF0) >> 4;
            uchar lo = hash[i] & 0x0F;
            if (hi >= mode.data1[0] && hi <= mode.data2[0]) { ++score; }
            if (lo >= mode.data1[0] && lo <= mode.data2[0]) { ++score; }
        }
        return score;

    case kLeadingRange:
        for (int i = 0; i < 20; ++i) {
            uchar hi = (hash[i] & 0xF0) >> 4;
            uchar lo = hash[i] & 0x0F;
            if (!(hi >= mode.data1[0] && hi <= mode.data2[0])) { return score; }
            ++score;
            if (!(lo >= mode.data1[0] && lo <= mode.data2[0])) { return score; }
            ++score;
        }
        return score;

    case kMirror:
        for (int i = 0; i < 10; ++i) {
            uchar leftLeft = (hash[9 - i] & 0xF0) >> 4;
            uchar leftRight = hash[9 - i] & 0x0F;
            uchar rightLeft = (hash[10 + i] & 0xF0) >> 4;
            uchar rightRight = hash[10 + i] & 0x0F;
            if (leftRight != rightLeft) { return score; }
            ++score;
            if (leftLeft != rightRight) { return score; }
            ++score;
        }
        return score;

    case kDoubles:
        for (int i = 0; i < 20; ++i) {
            // As in profanity.cl and now salt.cl: a byte's two nibbles are equal
            // exactly when the low four bits of (byte >> 4) ^ byte are clear,
            // which is one instruction rather than two masks and a compare.
            if ((((hash[i] >> 4) ^ hash[i]) & 0x0f) != 0) { return score; }
            ++score;
        }
        return score;

    default:
        return 0;
    }
}

/// Whether an address satisfies one --exact mask in full.
static bool matches_pattern(thread const uchar* hash, constant Pattern& pattern) {
    for (uint i = 0; i < 20; ++i) {
        // A wildcard has a zero mask byte and a zero want byte, so it compares
        // equal whatever the address holds there.
        if ((hash[i] & pattern.mask[i]) != pattern.want[i]) {
            return false;
        }
    }
    return true;
}
