# Correctness

Every failure mode in address mining is silent. A wrong pad byte, the wrong half of a hashed account, or nonce `0` instead of `1` still produces a well-formed 20-byte address. The miner reports a plausible hit, you deploy, and you land somewhere else — after paying for the GPU time. Nothing about the output tells you anything is wrong.

The same derivation is implemented up to four times here — OpenCL, Metal, Rust scalar, and the `sha3`-crate reference — so agreement has to be asserted rather than assumed.

## `--verify`, on by default

Every hit is re-derived on the CPU before it is printed. For salt modes the reported salt is pushed back through the derivation and compared to the reported address; for profanity the seed public key is walked forward by the reported offset and the resulting address compared.

This is the highest-value check in the project because it protects *real runs*, including on hardware and kernel variants nobody has tested. Hits are rare, so one CPU keccak per hit costs nothing measurable.

A hit that fails is printed with `[UNVERIFIED: the CPU does not agree this hit follows from its inputs]` and the process exits non-zero. That is a correctness bug, not a lucky find; please report it.

`--no-verify` turns it off. There is rarely a reason.

### What the re-derivation cannot say on its own

Both the salt and the address in a result slot come from the device, so comparing them establishes that the pair is self-consistent rather than that either is what the run asked for. In create2 and create3 the reported hit needs nothing further, because the salt *is* the result: one that derives the address printed beside it is usable whatever produced it.

What a self-consistent pair says nothing about is the score, and the score is not part of the result — it is the bar. Slots are indexed by score, so the slot a device writes into is its own account of what it found, and that number is published into an atomic every device reads back as `scoreMax`. Put a genuinely derived pair in a slot far above what the search is reaching and the bar rises for the whole rig, after which every later hit fails the kernel's own test and is never written: a rig that mines and reports nothing. So the reported address is scored again on the host, and a hit whose slot index disagrees comes back unverified and does not move the bar. Like the 1nft binding below, that check runs whatever `--no-verify` says, because it is a pass over 20 bytes rather than the per-hit re-derivation the flag exists to skip. profanity uses the same slot layout and gets the same treatment.

Withholding the bar is not specific to a wrong score. No hit the CPU failed to confirm raises it, for a reason that only shows up on a long run: the process bails on an unverified hit when the run ends, so one arriving in the first minute would otherwise have suppressed the following ten hours.

1nft is the exception, and it gets a second check. There the result is a magic, and the deployer rebuilds the salt around it from the account, so a salt whose low 16 bytes are not `keccak256(--mint-for)[16:32]` still derives a real address and still reports a magic — one that mints a different address than the one displayed. `SaltConfig::salt_binds_to_mint_for` reconstructs what the deployer would build, through the same `nft_salt` the mode is written in, and a hit that fails it is marked unverified like any other.

That one is **not** disabled by `--no-verify`. The flag exists to skip the per-hit re-derivation; the binding is a single keccak of 20 bytes, and a magic that cannot mint is malformed rather than merely unchecked. No honest device can trip it: the work-item words span salt bytes 3 to 14, so the pinned half comes back as it was sent. It is there for the salts that do not come from honest hardware, which on a rented machine is not a hypothetical — see [vastai.md](vastai.md).

## `1miner self-test`

Runs the checks against whatever device is actually present:

```bash
1miner self-test                  # default backend
1miner self-test --backend metal
1miner self-test --backend cpu
```

All four modes are covered, on every backend. The three salt modes get the planted target described below. profanity gets a short real search instead, in both plain and `--contract` shape, because a profanity work item's address cannot be predicted before the run the way a salt one can — so what is asserted is the property that protects a real run: every offset reported has to name the address reported with it.

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

**The secp256k1 kernel.** No target can be planted for profanity, so it is checked the way `self-test` checks it: a short search on a small round, with every offset the kernel reports required to name the address reported with it, in both plain and `--contract` shape. Then one of those offsets is handed to the CPU walk as its starting point. The kernel's field arithmetic is OpenCL C or Metal Shading Language and the host's is Rust, so that last step is two implementations agreeing on one scalar rather than one checked against itself — the claim the salt modes get from their planted targets. Run against both GPU backends.

That check is the whole safety net for one class of bug, because the two backends count rounds differently and nothing else would notice. OpenCL reads each round's results at the top of the next iteration, so its round counter lags its dispatches by one; Metal reads them immediately and has to add the one back. Get it wrong and every hit is still a real address with a well-formed offset — the offset simply names the key to a different address, which the operator discovers when the wallet opens on an empty account.

**Keccak variant equivalence.** Both permutations must find the same address for the same work item.

**1nft salt layout.** The reported magic plus the caller must rebuild exactly the salt that was mined, and the low 16 bytes must be the pinned caller digest.

**1nft account binding on the return path.** A result slot holding a salt with an unpinned low half, together with the address that salt really derives, has to come back unverified — with `--no-verify` as well, since the flag does not reach this check. The same slot built honestly has to come back verified, because a check that rejects a real hit costs a search.

**A device-chosen score on the return path.** A genuinely derived hit, planted in a slot index its address does not earn, has to come back unverified and leave the shared bar where it was, with `--no-verify` as well. Asserted for the salt slots and the profanity ones, which carry the same layout, and paired with the honest slot each time so that a check rejecting everything could not pass. A hit failing any other check has to withhold the bar too, which is asserted separately, since the two reasons take different routes to the same flag.

GPU-dependent tests skip rather than fail when no device is present, so the suite still runs in CI and inside a container.

## Why the CPU reference is built differently on purpose

`miner-core` does not copy the kernels' split-padding trick. It builds ordinary byte strings and hashes them with the `sha3` crate. Two implementations that share a shortcut also share its bugs; arriving at the same answer by different routes is what makes the comparison worth running.
