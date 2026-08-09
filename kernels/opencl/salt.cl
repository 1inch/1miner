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

__kernel void salt_iterate(__global result * const pResult, __global const mode * const pMode, const uchar scoreMax, const uint deviceIndex, const uint round) {
	ethhash h = { .q = { SALT_INITHASH } };
	SALT_APPLY_WORK_ITEM(h)

	// CREATE2: keccak(0xff ++ deployer ++ salt ++ codeHash), address at h.b[12:32].
	sha3_keccakf(&h);

#ifdef SALT_SECOND_HASH
	// CREATE3 second step: the CREATE2 result is a proxy which deploys at
	// nonce 1, so hash RLP([proxy, 1]) = 0xd6 0x94 ++ proxy ++ 0x01.
	ethhash h2 = { 0 };
	h2.b[0] = 0xd6;
	h2.b[1] = 0x94;
	for (int i = 0; i < 20; i++) {
		h2.b[2 + i] = h.b[12 + i];
	}
	h2.b[22] = 0x01;
	// Leading keccak pad bit for a 23-byte message; sha3_keccakf adds the 0x80.
	h2.b[23] ^= 0x01;
	sha3_keccakf(&h2);
	h = h2;
#endif

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

void salt_result_update(const uchar * const H, __global result * const pResult, const uchar score, const uchar scoreMax, const uint deviceIndex, const uint round) {
	if (score && score > scoreMax) {
		// One slot per score, first writer wins.
		const uchar hasResult = atomic_inc(&pResult[score].found);
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
