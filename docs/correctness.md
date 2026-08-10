# Correctness

Every failure mode in address mining is silent. A wrong pad byte, the wrong half of a hashed account, or nonce `0` instead of `1` still produces a well-formed 20-byte address. The miner reports a plausible hit, you deploy, and you land somewhere else — after paying for the GPU time. Nothing about the output tells you anything is wrong.

The same derivation is implemented up to four times here — OpenCL, Metal, Rust scalar, and the `sha3`-crate reference — so agreement has to be asserted rather than assumed.

## `--verify`, on by default

Every hit is re-derived on the CPU before it is printed. For salt modes the reported salt is pushed back through the derivation and compared to the reported address; for profanity the seed public key is walked forward by the reported offset and the resulting address compared.

This is the highest-value check in the project because it protects *real runs*, including on hardware and kernel variants nobody has tested. Hits are rare, so one CPU keccak per hit costs nothing measurable.

A hit that fails is printed with `[UNVERIFIED: CPU re-derivation disagrees with the kernel]` and the process exits non-zero. That is a correctness bug, not a lucky find; please report it.

`--no-verify` turns it off. There is rarely a reason.

## `1miner self-test`

Runs the checks against whatever device is actually present:

```bash
1miner self-test                  # default backend
1miner self-test --backend metal
1miner self-test --backend cpu
```

All four modes are covered. The three salt modes get the planted target described below. profanity gets a short real search instead, in both plain and `--contract` shape, because a profanity work item's address cannot be predicted before the run the way a salt one can — so what is asserted is the property that protects a real run: every offset reported has to name the address reported with it. Metal prints those two as `skip`, having no secp256k1 kernel to test.

Worth running on a freshly rented box before committing it to a long job. It takes a second or two and catches a bad driver or a miscompiled kernel before the rental clock has cost you anything.

## What the test suite covers

`cargo test` (add `--features metal` on macOS):

**Known-answer vectors.** The six official EIP-1014 CREATE2 examples. The CREATE3 vector `0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf` with a zero salt giving `0x6c8Ed9dC3734d7944BEDDd2fB5AcdF5f17247870`, which is independently cross-checked upstream in Rust, in Foundry, and by hand with `cast keccak`. The intermediate CREATE2 proxy is pinned separately, so a compensating error in both CREATE3 steps cannot hide. EIP-55 checksums. RLP nonce encoding, asserting that nonce 0 and nonce 1 differ.

**Constant guards.** `keccak256(0x67363d3d37363d34f03d5260086018f3)` must equal the default proxy code hash, so a corrupted constant cannot pass silently.

**The secp256k1 implementation** against the standard 2G and 3G vectors, the address of the generator, additive consistency between `a+b` and scalar multiplication, and the point at infinity. The generated 8160-entry precomputed table is compared entry by entry against profanity2's checked-in [`precomp.cpp`](https://github.com/1inch/profanity2/blob/master/src/precomp.cpp), which pins the field arithmetic and the table index layout together. That comparison reads the upstream tree from `references/`, which is gitignored, so it skips on a fresh clone and in every container build. A digest of the whole table is pinned beside it and runs everywhere; it catches a regression but cannot replace the cross-check, since it pins the table against itself rather than against profanity2.

**Scoring parity.** Each of the eight scoring functions is a line-by-line port of its kernel counterpart, including the break-on-first-miss behaviour that separates `leading` from `range`, with tests covering the boundaries.

**Cross-backend planted targets.** For each of create2, create3 and 1nft, a work item is chosen, its address computed on the CPU, and a full 20-byte mask built from it. The backend must find exactly that work item and report the matching salt. Passing exercises the pre-image construction, the keccak padding, the second CREATE hash, the scoring and the kernel's separate salt-reconstruction path all at once. Run against CPU, OpenCL and Metal.

**`--exact` returns a whole round.** The assertion is a count, not a presence: a mask of one nibble matches about one candidate in sixteen, so a round yields hundreds, and the test requires far more hits than the number of rounds. That is what separates the append-style buffer from the one-slot-per-score layout, which could only ever return one per round — and the earlier test, which asked only that matches kept arriving, passed against both. Where a round overflows the 256-match buffer, the overflow has to be reported. Run against CPU, OpenCL and Metal, and separately against the profanity kernel, whose exact path derives its addresses through secp256k1 rather than keccak alone.

**Several masks in one pass.** Three disjoint masks, with every hit required to satisfy the one it was reported against and all three required to have matched something. The second half is what catches a kernel that tests only the first mask.

**The secp256k1 kernel.** No target can be planted for profanity, so it is checked the way `self-test` checks it: a short search on a small round, with every offset the kernel reports required to name the address reported with it, in both plain and `--contract` shape. Then one of those offsets is handed to the CPU walk as its starting point. The kernel's field arithmetic is OpenCL C and the host's is Rust, so that last step is two implementations agreeing on one scalar rather than one checked against itself — the claim the salt modes get from their planted targets.

**Keccak variant equivalence.** Both permutations must find the same address for the same work item.

**1nft salt layout.** The reported magic plus the caller must rebuild exactly the salt that was mined, and the low 16 bytes must be the pinned caller digest.

GPU-dependent tests skip rather than fail when no device is present, so the suite still runs in CI and inside a container.

## Why the CPU reference is built differently on purpose

`miner-core` does not copy the kernels' split-padding trick. It builds ordinary byte strings and hashes them with the `sha3` crate. Two implementations that share a shortcut also share its bugs; arriving at the same answer by different routes is what makes the comparison worth running.
