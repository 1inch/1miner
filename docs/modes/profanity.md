# profanity: vanity account and contract addresses

Searches private-key offsets, so that `seed_key + offset` controls an address matching your pattern. You hand the miner a **public** key and it hands back an **offset**. The private key stays where you made it, before the run and after it.

## Safe by design

The original profanity derived its keypairs from a 32-bit seed, so every address it ever produced was brute-forceable, and years later funds were taken from people who had used it: [the disclosure is worth reading](https://blog.1inch.io/a-vulnerability-disclosed-in-profanity-an-ethereum-vanity-address-tool). profanity2, which these kernels descend from, did not fix the randomness — it removed the need for any, by never generating a key at all.

`1miner` keeps that property. It takes a seed public key, searches for a scalar offset whose address matches your pattern, and reports the offset. You add the offset to your seed private key afterwards, on your own machine. Nothing that runs on the GPU could produce a key even if it wanted to, which is what makes it safe to mine on hardware you do not control: a rented GPU, a friend's workstation, a container somebody else operates.

The private-key flag other miners have is absent here on purpose, and will stay absent.

## 1. Generate a seed keypair

On a machine you trust. `scripts/profanity-keygen.sh` makes one with openssl and writes nothing to disk:

```bash
eval "$(scripts/profanity-keygen.sh)"
```

That sets `$PROFANITY_PK`, the 64-hex seed private key, and `$PROFANITY_PUBKEY`, the 128-hex public key the miner takes, and prints both. Copy the private half somewhere safe before you start a long run: nothing stored it, and it leaves with the shell. `. scripts/profanity-keygen.sh` does the same by sourcing instead of by `eval`.

By hand it is two openssl calls:

```bash
openssl ecparam -genkey -name secp256k1 -noout | openssl ec -text -noout
```

The `priv:` block is the seed private key; the `pub:` block is the public key, whose leading `04` byte marks it as uncompressed and is not part of what the miner wants. Strip the colons, the whitespace and that byte, and you have the 128 hex characters for `--public-key`.

The DER-and-`sed` one-liner that circulates for this, including the one this page used to carry, is worth avoiding. openssl 3 prints an `EC-Parameters` line ahead of the DER and those `sed` patterns were written when it did not, so the label never appears and what you get instead is header, wrapper and private key run together: 128 hex characters, which is exactly the length of the public key it was meant to be. Paste that in as the seed key and every step afterwards still succeeds, on the wrong key, and the first thing to disagree with you is the wallet you finally import into. This is why `profanity-keygen.sh` compares two different openssl renderings rather than checking a length, and prints nothing at all when they disagree.

If you already have a key to use as the seed, `cast wallet public-key --raw-private-key 0x...` derives its public key. Run that where you trust the machine, since the private key has to be present for it.

## 2. Mine

```bash
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading 0
```

A hit looks like this:

```
  Time:     2s  Score:  8  GPU0  Offset: 0x00000000002d76a49f72887c449dbcaa169768e4f1f957e16de3deba9a3bfbef  Address: 0x0000000028f9357e9E3e0fB18cb43f0122585B90
```

Only an improvement on the best score so far is printed, so the output thins out as a run goes on. Nothing stops on its own: pass `--seconds` for a bounded run, or interrupt with Ctrl-C when you have what you want. Every hit is re-derived on the CPU before it appears, so a printed offset is a checked offset. `-z` is accepted for `--public-key`, and `-c` for `--contract`, as in profanity2.

What `Score` counts is not the same in every mode, which is worth knowing before concluding that a run is going badly — a `--matching dead` hit that matches all four digits scores 2, because the mask constrains two whole bytes:

| Mode | `Score` counts |
| --- | --- |
| `--leading`, `--leading-range`, `--mirror` | matching hex digits, stopping at the first miss |
| `--zeros`, `--letters`, `--numbers`, `--range` | matching hex digits anywhere in the address |
| `--matching`, `--trailing`, `--exact` | satisfied bytes of the mask, so at most 20 |
| `--zero-bytes`, `--leading-doubles` | whole bytes, anywhere and leading respectively |
| `--benchmark` | nothing; it never scores or reports |

### The longest run of one digit — `--leading`

```bash
# 0x00000... — as many leading zeros as it can find
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading 0

# 0xaaaaa...
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading a
```

The score is the length of the run, so the miner keeps finding better addresses and never finishes. Use it when you do not have a particular pattern in mind and just want something striking.

### A pattern in a known place — `--matching` and `--trailing`

`--matching` anchors a mask to the front of the address, `--trailing` to the back. Both take up to 40 hex characters, one per digit of the address, and any character that is not a hex digit is a wildcard. `X` is the convention; `.` reads better in a long mask.

```bash
# 0xdead...
1miner profanity --public-key "$PROFANITY_PUBKEY" --matching dead

# 0x...c0de
1miner profanity --public-key "$PROFANITY_PUBKEY" --trailing c0de

# 0x1111...2222 — both ends at once needs a full 40-character mask
1miner profanity --public-key "$PROFANITY_PUBKEY" --matching 1111XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX2222

# 0x????cafe... — digits 5 to 8 are "cafe" and the rest is free
1miner profanity --public-key "$PROFANITY_PUBKEY" --matching XXXXcafe
```

profanity2 has no `--trailing`: there a suffix is a 40-character mask with wildcards padding the front, which also works here.

Every fixed digit you add multiplies the expected search by 16. At 380 MH/s, the figure the [README](../../README.md) records for an M4 Max:

| Fixed digits | Expected attempts | Expected time |
| --- | --- | --- |
| 6 | 17 million | instant |
| 8 | 4.3 billion | 11 seconds |
| 10 | 1.1 trillion | 48 minutes |
| 12 | 280 trillion | 9 days |
| 14 | 72 quadrillion | 6 years |

Those are means, and the distribution has no memory: a run can take four times as long as the table says without anything being wrong. Half the pattern is often worth more than twice the wait.

Because only improvements are printed, once an address has satisfied every fixed position in the mask nothing better can appear, and the run is finished in every sense except stopping. When that is what you are waiting for, `--exact` is the better tool.

### Every match rather than the best one — `--exact`

`--exact` takes the same front-anchored mask, with the same wildcards, and drops the scoring: it reports every address that satisfies the whole mask and keeps going.

```bash
# every 0x1337c0de... it finds, not just the first
1miner profanity --public-key "$PROFANITY_PUBKEY" --exact 1337c0de

# every address with five or more leading zeros
1miner profanity --public-key "$PROFANITY_PUBKEY" --exact 00000

# every 0x...beef, since --exact is front-anchored like --matching
1miner profanity --public-key "$PROFANITY_PUBKEY" --exact XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbeef
```

It says at the start how many bytes have to match, and the score column then stops moving, because there is only one score worth having. Expect the first result to take longer to appear than with `--matching`, which reports its near misses on the way; here there are no near misses. This is the mode to use when you want several candidates to pick between rather than one.

A device reports at most one match per round, and a round is about 4.2 million candidates on the default geometry. A mask of six digits or more comes up less often than that, so nothing is lost. Below it most of the matches are simply never shown, which is one more reason not to mine a pattern you would hit by accident.

### Character classes anywhere — `--zeros`, `--letters`, `--numbers`

```bash
# as many 0 digits as possible, anywhere: 0x00aa0e050bb03c0b066e00c0f70a03d0d000b0d7
1miner profanity --public-key "$PROFANITY_PUBKEY" --zeros

# nothing but a-f: 0xffcadbfaecfcdeaddeedabadfeedfacebeefcafe
1miner profanity --public-key "$PROFANITY_PUBKEY" --letters

# nothing but 0-9: 0x8896339129744478701529940603494328137361
1miner profanity --public-key "$PROFANITY_PUBKEY" --numbers
```

### Ranges — `--leading-range` and `--range`

`-m`/`--min` and `-M`/`--max` give the bounds as nibble values, 0 for `0` and 15 for `f`, defaulting to the whole range of 0 to 15. They belong to these two modes alone, and are rejected beside any other scoring flag rather than being accepted and ignored:

```bash
# leading digits all 0 or 1: 0x0110100...
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading-range -m 0 -M 1

# leading digits all a, b or c: 0xcbabacc...
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading-range -m 10 -M 12

# digits 0 or 1 anywhere in the address
1miner profanity --public-key "$PROFANITY_PUBKEY" --range -m 0 -M 1
```

`--zeros`, `--letters` and `--numbers` are `--range` with the bounds filled in, at 0-0, a-f and 0-9.

### Other shapes — `--mirror`, `--leading-doubles`, `--zero-bytes`

```bash
# a palindrome measured out from the middle: 0x...abccba...
1miner profanity --public-key "$PROFANITY_PUBKEY" --mirror

# leading pairs of identical digits: 0x00ffcc55...
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading-doubles

# as many whole zero bytes as possible: 0x00815e00c0fd4a2d00ae00fa00e300ee00fc0034
1miner profanity --public-key "$PROFANITY_PUBKEY" --zero-bytes
```

`--zero-bytes` is the one of the three with a reason beyond appearance: a zero byte in calldata is charged 4 gas rather than 16, so an address full of them is cheaper for everyone who passes it as an argument.

### A vanity contract address — `--contract`

`--contract` combines with any scoring mode. The score then applies to the contract the found account would deploy, rather than to the account address itself:

```bash
# an account whose first deployment lands on 0x00000...
1miner profanity --public-key "$PROFANITY_PUBKEY" --contract --leading 0

# and whose first deployment has the most zero bytes
1miner profanity --public-key "$PROFANITY_PUBKEY" --contract --zero-bytes
```

#### It assumes the account has never sent a transaction

A CREATE address is derived from the deploying account and its nonce, and `--contract` scores the address for **nonce 0**. The contract only lands where the miner said if the deployment is that account's very first transaction.

Send anything else from it first — a test transfer, an approval, a failed attempt — and the nonce has moved on, so the mined address is gone for good. Fund the account, deploy, and do nothing else with it in between. What an account would deploy to at any nonce is one command:

```bash
cast compute-address <ADDRESS> --nonce 0
```

### Devices, tuning and a plain benchmark

```bash
# throughput with no scoring at all
1miner profanity --public-key "$PROFANITY_PUBKEY" --benchmark --seconds 30

# every GPU is used by default; skip one by index, an integrated device say
1miner profanity --public-key "$PROFANITY_PUBKEY" --leading 0 --skip 1
```

`-i`/`--inverse-size` and `-I`/`--inverse-multiple` set the batch geometry. Their product is the number of points per round, and it dominates both memory use and start-up time: the defaults of 255 and 16384 are about 4.2M points and roughly 400 MB of device memory. Lower `-I` first if a device runs out of memory or takes too long to start. [backends.md](../backends.md) has the rest of the tuning flags, and [benchmarking.md](../benchmarking.md) explains why a single back-to-back run is not a measurement.

## 3. Turn the offset into the private key

```bash
scripts/profanity-final-key.sh \
  --offset 0x00000000002d76a49f72887c449dbcaa169768e4f1f957e16de3deba9a3bfbef \
  --address 0x0000000028f9357e9E3e0fB18cb43f0122585B90
```

The script reads the seed key from `$PROFANITY_PK`, or takes it as an argument, adds the offset modulo the group order, then derives the address the result actually controls and fails unless it is the one you passed as `--address`. The key is the only thing on stdout, so `key=$(scripts/profanity-final-key.sh --offset 0x...)` is a usable idiom.

By hand it is one addition:

```
final_private_key = (seed_private_key + offset) mod n
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
```

```python
seed = 0x0bc657b0af28b743c7f0d49c4de78efd47a5c8923dabfdef051fff5cdc7c30e7
offset = 0x0000f8ba428990fca1e618a252ac3614f5de19b20ff00c2ded57bfb6933830aa
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
print("%064x" % ((seed + offset) % n))
```

The `%064x` is the part to copy. About one sum in sixteen is shorter than 64 characters, and neither Python's `hex()` nor `bc` prints the leading zeros — a private key one digit short is a different key, and what a wallet does with it is up to the wallet:

```
seed private key   0bc657b0af28b743c7f0d49c4de78efd47a5c8923dabfdef051fff5cdc7c30e7
offset             0000f8ba428990fca1e618a252ac3614f5de19b20ff00c2ded57bfb6933830aa
the sum, 63 long    bc7506af1b2484069d6ed3ea093c5123d83e2444d9c0a1cf277bf136fb46191
private key        0bc7506af1b2484069d6ed3ea093c5123d83e2444d9c0a1cf277bf136fb46191
```

Do this offline, and never in an online calculator: a page that adds two numbers for you is a page you have handed a private key to.

## 4. Verify before you fund it

Import the key into the wallet you intend to use and confirm it shows the address the miner reported. Two things have already agreed by this point — the miner re-derives every hit on the CPU before printing it, and `--address` re-derives the address from the finished key — but neither of them is the wallet, and this step costs a minute. Do it before sending anything of value.

## Notes

- The offset's top 16 bits are always zero, so adding it to a seed key cannot overflow 256 bits.
- Every run starts at a random offset, on `--backend cpu` as much as on a GPU, so searching the same public key again covers new ground instead of repeating itself.
- Each GPU gets its own slice of the offset space rather than a random start of its own, so no two devices in a run can cover the same ground.
- `1miner self-test` mines this mode briefly on your own device and re-derives every hit, with and without `--contract`. It is the same check `--verify` makes during a run, asked before you commit to one rather than hours in.
- Metal does not support this mode: it needs a secp256k1 kernel, and only the keccak-based salt modes have one. Use `--backend opencl`.
- `--backend cpu` works and is useful for checking, but it performs a modular inversion per step and is orders of magnitude slower.
- The kernel deliberately skips the equal-x edge cases in point addition, as the reference implementation does. They are astronomically rare and cost only a missed candidate.
- `--no-verify` turns off the CPU re-derivation of each hit. Nothing here is fast enough to make that worthwhile.
