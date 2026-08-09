# Backends and kernels

A backend decides *where* candidates are enumerated. A kernel decides *how*. Both are chosen at run time so you can race implementations against each other on your own hardware rather than trusting someone else's benchmark.

## Backends

| `--backend` | Modes | Notes |
| --- | --- | --- |
| `opencl` (default) | all four | The only backend that supports `profanity`. Multi-GPU. |
| `metal` | create2, create3, 1nft | macOS only, requires `--features metal` at build time. |
| `cpu` | all four | Portable fallback and the reference the others are checked against. Two-lane NEON on aarch64. |

`profanity` is OpenCL-only because it needs secp256k1 field arithmetic on the GPU, and only the OpenCL kernel has it. The Metal path covers the keccak-based salt modes. CUDA is the intended next backend and slots in the same way.

Metal exists because Apple deprecated OpenCL: on Apple silicon it is capped at OpenCL 1.2 and is not the fast path the driver is tuned for. On an M4 Max the two are close, with Metal slightly ahead for create3.

## The CPU backend

Its first job is to be the reference the accelerated backends are checked against, but it is a usable miner for small searches and the answer on a machine with no GPU at all.

On aarch64 the salt modes use a two-lane NEON Keccak, holding each of the 25 Keccak lanes as a `uint64x2_t` so one permutation covers two candidates. That is worth 2.3x on an M4 Max: 88.7 against 38.0 MH/s for create3. It is still several times slower than the same machine's GPU, so this is a fallback rather than a contender.

`MINER_NO_NEON=1` forces the plain scalar path. Use it to A/B the two, or as a safety valve if the SIMD path ever misbehaves on some hardware. The test suite asserts both produce identical addresses for every mode, so the choice is a performance one only.

The ARMv8.2 SHA3 extension would be faster still, but its Rust intrinsics are not yet stable, so it is not used.

## Kernels

`--kernel` selects the Keccak permutation used by the OpenCL backend:

- `tuned` (default) — the ERADICATE2/3 permutation: fused theta with five temporaries and a rotated chi step.
- `plain` — profanity2's more literal permutation.

They are functionally identical, which the test suite asserts by having both find the same planted address. On an Apple M4 Max they measure within noise of each other, because the driver's compiler optimises both to much the same code. On other hardware the gap may be real, which is the point of being able to switch.

## Adding a kernel variant

1. Put the source in `kernels/opencl/` (or `kernels/metal/`).
2. Add a `const` for it in `crates/miner-backend/src/kernels.rs`, which embeds sources with `include_str!` so the binary stays self-contained.
3. Add a variant to `KeccakVariant` in `crates/miner-backend/src/lib.rs`, wiring up `source()`, `as_str()`, `parse()` and `all()`.
4. Add the name to the `--kernel` value list in `crates/miner-cli/src/cli.rs`.

The interface a Keccak source must provide is small: the `ethhash` union and `void sha3_keccakf(ethhash *)`. Note that `sha3_keccakf` is expected to apply the trailing `0x80` pad byte itself; see [how-address-derivation-works.md](how-address-derivation-works.md).

Then check it and time it:

```bash
cargo test --test derivation          # must still agree with the CPU
1miner self-test --kernel mynew       # and on your actual GPU
1miner create3 --deployer 0x... --benchmark --kernel mynew --seconds 30
```

A new kernel that is fast and wrong is the failure this project is arranged to prevent, so run the agreement tests before you believe the number.

## Tuning

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `-w`, `--work` | all GPU backends | Local work size / threadgroup width. `0` lets the driver choose. |
| `-W`, `--work-max` | OpenCL | Largest single enqueue; rounds are split into chunks of this size. |
| `-S`, `--size` | salt modes | Candidates per round per device. Default 16777216. |
| `-i`, `--inverse-size` | profanity | Batched-inversion width. Default 255. |
| `-I`, `--inverse-multiple` | profanity | Parallel inverse batches. Default 16384. |
| `-s`, `--skip` | OpenCL | Skip a device by index. Repeatable. |
| `-n`, `--no-cache` | OpenCL | Ignore the compiled-kernel cache. |
| `--threads` | cpu | Worker threads. Defaults to available parallelism. |

For profanity, `--inverse-size` times `--inverse-multiple` is both the number of points per round and the driver of memory use: three scratch buffers of 32 bytes per point, so the default 4.2M points needs roughly 400 MB. Lower `-I` first if a device runs out of memory or takes too long to initialise.
