/* Metal salt-search kernel for create2, create3 and 1nft.
 *
 * Mirrors kernels/opencl/salt.cl. The Keccak-f permutation and the scoring
 * functions come from keccak.metal and scoring.metal, prepended to this source
 * by the host, and the permutation applies the trailing 0x80 pad byte itself
 * while callers supply the leading 0x01 bit.
 *
 * Unlike the OpenCL path, the 200-byte pre-image arrives in a buffer rather
 * than as a compile-time constant, so changing deployer, code hash or base
 * salt does not force a pipeline rebuild.
 */

struct Params {
    ulong state[25];   // CREATE2 pre-image plus the leading keccak pad bit
    uint  deviceIndex;
    uint  round;
    // Added to thread_position_in_grid. Metal has no equivalent of OpenCL's
    // global work offset, so without this a round split into chunks would start
    // every chunk's ids at zero and mine the same salts over again.
    uint  idBase;
    uint  secondHash;  // 1 for create3 and 1nft
    uint  scoreMax;
    uint  patternCount;   // --exact only
    uint  exactCapacity;  // --exact only: highest slot salt_iterate_exact fills
};

struct Result {
    uchar salt[32];
    uchar hash[20];
    uint  found;
};

/// Apply this work item's coordinates, matching SALT_APPLY_WORK_ITEM in the
/// OpenCL kernel and SaltConfig::salt_at on the host.
static void apply_work_item(thread ulong* state, uint deviceIndex, uint gid, uint round) {
    // Words 6, 7 and 8 as 32-bit little-endian views of the 200-byte state.
    uint w6 = (uint)((state[3] >> 0) & 0xffffffffUL);
    uint w7 = (uint)((state[3] >> 32) & 0xffffffffUL);
    uint w8 = (uint)((state[4] >> 0) & 0xffffffffUL);

    w6 += deviceIndex;
    w7 += gid;
    w8 += round;

    state[3] = ((ulong)w7 << 32) | (ulong)w6;
    state[4] = (state[4] & 0xffffffff00000000UL) | (ulong)w8;
}

/// The salt this work item tries and the address it derives. Shared by both
/// kernels below so neither can drift in what it hashes.
static void salt_derive(
    constant Params& params,
    uint gid,
    thread uchar* salt,
    thread uchar* address)
{
    ulong state[25];
    for (int i = 0; i < 25; ++i) {
        state[i] = params.state[i];
    }
    apply_work_item(state, params.deviceIndex, gid, params.round);

    // Keep the salt before the permutation destroys the state.
    for (uint i = 0; i < 32; ++i) {
        salt[i] = byte_at(state, i + 21);
    }

    keccakf(state);

    for (uint i = 0; i < 20; ++i) {
        address[i] = byte_at(state, i + 12);
    }

    if (params.secondHash != 0) {
        ulong second[25];
        for (int i = 0; i < 25; ++i) {
            second[i] = 0;
        }
        set_byte(second, 0, 0xd6);
        set_byte(second, 1, 0x94);
        for (uint i = 0; i < 20; ++i) {
            set_byte(second, 2 + i, address[i]);
        }
        set_byte(second, 22, 0x01);
        set_byte(second, 23, 0x01); // leading keccak pad bit
        keccakf(second);
        for (uint i = 0; i < 20; ++i) {
            address[i] = byte_at(second, i + 12);
        }
    }
}

kernel void salt_iterate(
    device Result* results        [[buffer(0)]],
    constant Mode& mode           [[buffer(1)]],
    constant Params& params       [[buffer(2)]],
    device atomic_uint* foundFlags [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]])
{
    uchar salt[32];
    uchar address[20];
    salt_derive(params, gid + params.idBase, salt, address);

    int score = score_address(address, mode);
    if (score <= 0 || (uint)score <= params.scoreMax) {
        return;
    }

    // One slot per score, first writer wins.
    if (atomic_fetch_add_explicit(&foundFlags[score], 1u, memory_order_relaxed) != 0) {
        return;
    }
    for (uint i = 0; i < 32; ++i) {
        results[score].salt[i] = salt[i];
    }
    for (uint i = 0; i < 20; ++i) {
        results[score].hash[i] = address[i];
    }
    results[score].found = 1;
}

/// --exact: every address matching one of the masks in full, appended in
/// arrival order rather than kept one per score.
///
/// A different question from salt_iterate and so a different buffer. There is
/// no score and no bar: a candidate either matches completely or is not a
/// result, and all of them are wanted, which one slot per score cannot express.
///
/// foundFlags[0] counts this round's matches and results[1..exactCapacity] hold
/// them. The host clears the counter before each round, so the bounds check is
/// also what keeps two threads out of one slot. Matches past the capacity are
/// counted but not stored, and the host reports how many. Which mask matched
/// goes in the slot's own `found`, unused here since only slot 0 counts.
kernel void salt_iterate_exact(
    device Result* results         [[buffer(0)]],
    constant Pattern* patterns     [[buffer(1)]],
    constant Params& params        [[buffer(2)]],
    device atomic_uint* foundFlags [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]])
{
    uchar salt[32];
    uchar address[20];
    salt_derive(params, gid + params.idBase, salt, address);

    for (uint p = 0; p < params.patternCount; ++p) {
        if (!matches_pattern(address, patterns[p])) {
            continue;
        }

        uint slot = atomic_fetch_add_explicit(&foundFlags[0], 1u, memory_order_relaxed) + 1;
        if (slot <= params.exactCapacity) {
            for (uint i = 0; i < 32; ++i) {
                results[slot].salt[i] = salt[i];
            }
            for (uint i = 0; i < 20; ++i) {
                results[slot].hash[i] = address[i];
            }
            results[slot].found = p + 1;
        }
        // One address can satisfy two masks; report it once, against the first.
        return;
    }
}
