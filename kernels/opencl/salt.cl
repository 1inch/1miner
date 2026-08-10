/* Unified salt-search kernel for create2, create3 and 1nft.
 *
 * Derived from ERADICATE2 and ERADICATE3 by Johan Gustafsson, which are the
 * same program apart from one extra keccak. That difference is a compile-time
 * switch here (SALT_SECOND_HASH) so all three modes share one kernel.
 *
 * Requires a keccak implementation providing `ethhash` and `sha3_keccakf`,
 * prepended by the host. Note that `sha3_keccakf` applies the trailing 0x80
 * keccak pad byte itself; callers supply only the leading 0x01 pad bit.
 *
 * Defines expected from the host:
 *   SALT_INITHASH     25 comma-separated ulongs: the 200-byte keccak state
 *                     holding 0xff ++ deployer ++ salt ++ codeHash and the pad.
 *   SALT_MAX_SCORE    highest score slot in the result buffer.
 *   SALT_SECOND_HASH  defined for create3 and 1nft, absent for create2.
 */

enum ModeFunction {
	Benchmark, ZeroBytes, Matching, Leading, Range, Mirror, Doubles, LeadingRange
};

typedef struct {
	enum ModeFunction function;
	uchar data1[20];
	uchar data2[20];
} mode;

typedef struct __attribute__((packed)) {
	uchar salt[32];
	uchar hash[20];
	uint found;
} result;

/* One --exact mask. `mask` has 0xF nibbles where a digit was given and 0 where
 * it was a wildcard, `want` the digits themselves, so a candidate matches when
 * (address[i] & mask[i]) == want[i] for all twenty bytes. */
typedef struct __attribute__((packed)) {
	uchar mask[20];
	uchar want[20];
} pattern;

void salt_result_update(const uchar * const hash, __global result * const pResult, const uchar score, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_benchmark(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_zerobytes(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_matching(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_leading(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_range(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_leadingrange(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_mirror(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);
void salt_score_doubles(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round);

/* The salt occupies h.b[21:52], spanning words h.d[6:12]. Three of those words
 * are bumped to give every device, work-item and round a distinct salt.
 * Overflow is ignored: at the default 2**24 work-items a device would need
 * 2**56 attempts before repeating.
 *
 * Keep this in sync with SaltConfig::salt_at on the host, which reproduces the
 * same arithmetic to verify reported hits. */
#define SALT_APPLY_WORK_ITEM(h)          \
	h.d[6] += deviceIndex;               \
	h.d[7] += get_global_id(0);          \
	h.d[8] += round;

#ifdef SALT_SECOND_HASH
/* CREATE3 second step: the CREATE2 result is a proxy which deploys at nonce 1,
 * so hash RLP([proxy, 1]) = 0xd6 0x94 ++ proxy ++ 0x01.
 *
 * The temporary has a fixed name rather than one pasted from the argument:
 * `h##2.b` pastes `h` onto `2.b`, which lexes as a floating-point constant and
 * is not a valid token to form. */
#define SALT_APPLY_SECOND_HASH(h)             \
	ethhash hSecond = { 0 };                  \
	hSecond.b[0] = 0xd6;                      \
	hSecond.b[1] = 0x94;                      \
	for (int i = 0; i < 20; i++) {            \
		hSecond.b[2 + i] = h.b[12 + i];       \
	}                                         \
	hSecond.b[22] = 0x01;                     \
	/* Leading pad bit for 23 bytes; sha3_keccakf adds the 0x80. */ \
	hSecond.b[23] ^= 0x01;                    \
	sha3_keccakf(&hSecond);                   \
	h = hSecond;
#else
#define SALT_APPLY_SECOND_HASH(h)
#endif

/* The whole derivation for this work item, leaving the address at h.b[12:32].
 *
 * A macro rather than a function so that the scoring kernel below generates
 * exactly what it did before this was shared: a function would either copy the
 * twenty address bytes out or return a pointer into a local, and the scorers
 * read straight out of the state. The exact kernel uses the same text, so the
 * two cannot drift in what they hash. */
#define SALT_DERIVE(h)                                                    \
	/* CREATE2: keccak(0xff ++ deployer ++ salt ++ codeHash). */          \
	ethhash h = { .q = { SALT_INITHASH } };                               \
	SALT_APPLY_WORK_ITEM(h)                                               \
	sha3_keccakf(&h);                                                     \
	SALT_APPLY_SECOND_HASH(h)

__kernel void salt_iterate(__global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	SALT_DERIVE(h)

	switch (pMode->function) {
	case Benchmark:
		salt_score_benchmark(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case ZeroBytes:
		salt_score_zerobytes(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case Matching:
		salt_score_matching(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case Leading:
		salt_score_leading(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case Range:
		salt_score_range(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case Mirror:
		salt_score_mirror(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case Doubles:
		salt_score_doubles(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	case LeadingRange:
		salt_score_leadingrange(h.b + 12, pResult, pMode, scoreMax, deviceIndex, round);
		break;
	}
}

/* --exact: every address matching one of the masks in full, appended in arrival
 * order rather than kept one per score.
 *
 * This asks a different question from salt_iterate and so keeps a different
 * buffer. There is no score and no bar: a candidate either matches a mask
 * completely or it is not a result at all, and every match is wanted, which the
 * one-slot-per-score layout cannot express.
 *
 * pResult[0].found counts the matches this round found; they are stored at
 * pResult[1..SALT_EXACT_CAPACITY]. The host clears the counter before each
 * round and drains what it finds, so the bounds check below is also what
 * guarantees no two work items claim the same slot. Matches past the capacity
 * are counted but not stored, and the host reports how many.
 *
 * A match records which mask it matched in its own `found` field, which is
 * unused in this layout because only slot 0 counts. */
__kernel void salt_iterate_exact(__global result * const pResult, __global const pattern * const pPatterns, const uint patternCount, const uint deviceIndex, const uint round) {
	SALT_DERIVE(h)

	for (uint p = 0; p < patternCount; ++p) {
		bool matched = true;
		for (int i = 0; i < 20; ++i) {
			if ((h.b[12 + i] & pPatterns[p].mask[i]) != pPatterns[p].want[i]) {
				matched = false;
				break;
			}
		}
		if (!matched) {
			continue;
		}

		const uint slot = atomic_inc(&pResult[0].found) + 1;
		if (slot <= SALT_EXACT_CAPACITY) {
			// Rebuild the state to recover the salt, as the scoring path does,
			// rather than carrying it through the hash: the host re-derives
			// every hit, and that only proves something while the two are
			// separate pieces of arithmetic.
			ethhash s = { .q = { SALT_INITHASH } };
			SALT_APPLY_WORK_ITEM(s)

			for (int i = 0; i < 32; ++i) {
				pResult[slot].salt[i] = s.b[i + 21];
			}
			for (int i = 0; i < 20; ++i) {
				pResult[slot].hash[i] = h.b[12 + i];
			}
			pResult[slot].found = p + 1;
		}
		// One address can satisfy two masks; report it once, against the first.
		return;
	}
}

void salt_result_update(const uchar * const H, __global result * const pResult, const uchar score, const uchar scoreMax, const uint deviceIndex, const uint round) {
	if (score && score > scoreMax) {
		// One slot per score, first writer wins. The counter is a uint, and
		// truncating it here would let every 256th writer believe it was first;
		// two that do interleave their writes, leaving one item's salt beside
		// another's address.
		const uint hasResult = atomic_inc(&pResult[score].found);
		if (hasResult == 0) {
			// Rebuild this work-item's state to recover the salt that produced
			// the hit. This repeats the arithmetic above rather than carrying
			// the salt through, so the host re-derives every hit to confirm the
			// two agree.
			ethhash h = { .q = { SALT_INITHASH } };
			SALT_APPLY_WORK_ITEM(h)

			for (int i = 0; i < 32; ++i) {
				pResult[score].salt[i] = h.b[i + 21];
			}
			for (int i = 0; i < 20; ++i) {
				pResult[score].hash[i] = H[i];
			}
		}
	}
}

/* Returning a constant 0 is safe here, and deliberately unlike profanity.cl's
 * benchmark scorer, which has to consume the address bytes. This kernel picks
 * its scorer with a runtime switch on a value read from pMode, so the compiler
 * must keep every branch and the hash above stays live whatever this one does.
 * profanity.cl selects at compile time through PROFANITY_SCORE_KERNEL, where a
 * constant would let the keccak behind it be eliminated and the reported
 * hashrate become fiction. Do not "fix" the asymmetry in either direction. */
void salt_score_benchmark(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	salt_result_update(hash, pResult, 0, scoreMax, deviceIndex, round);
}

void salt_score_zerobytes(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		score += !hash[i];
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_matching(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		if (pMode->data1[i] > 0 && (hash[i] & pMode->data1[i]) == pMode->data2[i]) {
			++score;
		}
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_leading(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		if (((hash[i] & 0xF0) >> 4) == pMode->data1[0]) {
			++score;
		} else {
			break;
		}

		if ((hash[i] & 0x0F) == pMode->data1[0]) {
			++score;
		} else {
			break;
		}
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_range(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		const uchar first = (hash[i] & 0xF0) >> 4;
		const uchar second = (hash[i] & 0x0F);

		if (first >= pMode->data1[0] && first <= pMode->data2[0]) {
			++score;
		}
		if (second >= pMode->data1[0] && second <= pMode->data2[0]) {
			++score;
		}
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_leadingrange(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		const uchar first = (hash[i] & 0xF0) >> 4;
		const uchar second = (hash[i] & 0x0F);

		if (first >= pMode->data1[0] && first <= pMode->data2[0]) {
			++score;
		} else {
			break;
		}

		if (second >= pMode->data1[0] && second <= pMode->data2[0]) {
			++score;
		} else {
			break;
		}
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_mirror(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 10; ++i) {
		const uchar leftLeft = (hash[9 - i] & 0xF0) >> 4;
		const uchar leftRight = (hash[9 - i] & 0x0F);

		const uchar rightLeft = (hash[10 + i] & 0xF0) >> 4;
		const uchar rightRight = (hash[10 + i] & 0x0F);

		if (leftRight != rightLeft) {
			break;
		}
		++score;

		if (leftLeft != rightRight) {
			break;
		}
		++score;
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}

void salt_score_doubles(const uchar * const hash, __global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	int score = 0;
	for (int i = 0; i < 20; ++i) {
		if (((hash[i] & 0xF0) >> 4) == (hash[i] & 0x0F)) {
			++score;
		} else {
			break;
		}
	}
	salt_result_update(hash, pResult, score, scoreMax, deviceIndex, round);
}
