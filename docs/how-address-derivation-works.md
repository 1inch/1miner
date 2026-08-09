# How address derivation works

All four modes end the same way: hash something, take the low 20 bytes. What differs is the pre-image. This page collects the details that are easy to get subtly wrong, because **every mistake here is silent** — a wrong pad byte or the wrong RLP nonce still yields a well-formed address, and you find out only when you deploy and land somewhere else.

## The four derivations

```
EOA       keccak256(pubkey_x ++ pubkey_y)[12:]
CREATE    keccak256(rlp([sender, nonce]))[12:]
CREATE2   keccak256(0xff ++ deployer ++ salt ++ keccak256(initCode))[12:]
CREATE3   proxy = CREATE2(factory, salt, keccak256(proxyBytecode))
          address = CREATE(proxy, 1)
```

CREATE3 is two steps: a minimal proxy is deployed with CREATE2, and the proxy then deploys the real contract with CREATE at nonce 1. The consequence people actually care about is that a CREATE3 address depends only on the factory and the salt, **not** on the init code, so you can mine an address before the contract is written.

The proxy is `0x67363d3d37363d34f03d5260086018f3` and its hash is `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`. A unit test asserts the second follows from the first rather than trusting the constant.

## RLP nonces: 0 is not 0x00

For a 20-byte sender and a nonce below 0x80, `rlp([sender, nonce])` is 23 bytes:

```
0xd6 0x94 ++ sender(20) ++ nonce
```

The trap is that RLP encodes a **zero** nonce as the empty string `0x80`, not as `0x00`. So:

- profanity's `--contract` scores the contract at nonce 0, encoding `0x80`.
- CREATE3's second step is at nonce **1**, encoding `0x01`.

Mixing these up shifts every result by a whole address.

## Keccak padding is split in two places

This is the single most likely thing to break in a port. The GPU kernels do not apply the full keccak padding in one place:

- **The caller** sets the *leading* pad bit. For the 85-byte CREATE2 pre-image that is `state[85] ^= 0x01`; for the 23-byte CREATE pre-image it is `h2.b[23] ^= 0x01`.
- **The permutation** applies the *trailing* `0x80`, unconditionally, as `h->d[33] ^= 0x80000000` in OpenCL and `st[16] ^= 0x80 << 56` in Metal. Byte 135 is the last byte of the keccak rate.

That makes `sha3_keccakf` valid for single-block messages only, which is all these pre-images need. It also means you cannot drop in a general keccak implementation without accounting for both halves. ERADICATE3's `// IDK why but it works` comment refers to exactly this.

`miner-core` deliberately does not copy the trick: it builds ordinary byte strings and hashes them with the `sha3` crate, so the CPU reference and the GPU arrive at the same answer by different routes. That is what makes agreement between them meaningful.

## What actually varies per work item

The salt occupies bytes 21..53 of the keccak state, which spans 32-bit words 6 through 12. Only three of those words move:

```
h.d[6] += deviceIndex
h.d[7] += get_global_id(0)
h.d[8] += round
```

Everything else, including the rest of the salt, is fixed for the run and comes from the randomised base salt. With the default 2^24 work items a device would need about 2^56 attempts before repeating, so the fixed remainder costs nothing in practice — but the base salt must be re-randomised every run, which the CLI does.

Two consequences worth knowing:

- The kernel **reconstructs** the salt in its result path, repeating this arithmetic rather than carrying the salt through the hash. If the two ever drift, the miner reports a salt that does not reproduce its own address. `SaltConfig::salt_at` mirrors it on the host and `--verify` checks every hit.
- Changing the deployer, code hash or base salt changes a compile-time constant in the OpenCL kernel, so the program is rebuilt. The Metal path passes the state in a buffer instead and does not rebuild.

## The 1nft salt layout

The 1inch Address NFT deployer derives its salt from the magic you supply and the account being minted for:

```
salt = magic(16) ++ keccak256(account)[16:32]
```

So only the high 16 bytes are searched; the low half is pinned before mining starts. The reported result is the **magic**, which is what the deployer's mint call takes. An address mined for one account is worthless for another, which is why `--mint-for` is required.

## Profanity: offsets, not keys

Profanity does not hash a salt. It walks elliptic curve points: each work item starts at `seed_pub + (seed + (id << 192)) * G` and every round advances all points by one generator step. A hit at round `r` in work item `id` therefore corresponds to the scalar

```
offset = seed + r + (id << 192)
```

and the miner prints that offset. Adding it to your seed private key modulo the curve order gives the private key for the found address. The miner only ever sees a public key.

`seed` is 192 random bits below a most significant lane set to `device_index << 32`. That reserves the 32 bits under the device's slot for `id`, so two GPUs in one run cannot produce the same offset however closely they start, and it leaves the top 16 bits clear so that adding the offset to a seed private key cannot overflow 256 bits.
