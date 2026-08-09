# Troubleshooting

## "no compute devices found for backend opencl"

The ICD loader found no GPU. Confirm independently:

```bash
clinfo -l
```

If `clinfo` also shows nothing, the problem is below 1miner.

- **Driver runtime missing.** Building needs headers; running needs your vendor's runtime. See [build/linux.md](build/linux.md).
- **In Docker without `--gpus all`.** The container has no GPU at all.
- **In Docker with `--gpus all` but still nothing.** The NVIDIA toolkit mounts the OpenCL library without registering it with the ICD loader. Our image writes `/etc/OpenCL/vendors/nvidia.icd` for exactly this reason; a custom image needs the same. See [docker.md](docker.md).
- **Sandboxed shell on macOS.** Some sandboxes block GPU access and OpenCL then reports zero devices on a machine that obviously has a GPU. Try an ordinary terminal.

## A kernel change had no effect

Compiled OpenCL binaries are cached, keyed on source, build options and device name, so an edited kernel should produce a new key. If you suspect the cache:

```bash
1miner create3 ... --no-cache
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/1miner/opencl"
```

## `[UNVERIFIED: CPU re-derivation disagrees with the kernel]`

The GPU reported an address that the CPU cannot reproduce from the reported salt or offset. This is a correctness bug, not a near miss: the result is not usable. The process exits non-zero deliberately.

Please report it with the backend, kernel variant, GPU and driver version. Then narrow it down:

```bash
1miner self-test --backend opencl
1miner self-test --backend cpu
cargo test --test derivation
```

If the CPU self-test passes and the GPU one fails, suspect the device or driver. Trying `--kernel plain` is a quick way to tell a kernel bug from a driver bug.

## `CL_INVALID_WORK_GROUP_SIZE` or a hang at launch

The local work size does not suit the device. Let the driver choose:

```bash
1miner create3 ... --work 0
```

## Out of memory, or profanity takes forever to start

Profanity allocates three buffers of `--inverse-size` times `--inverse-multiple` points, 32 bytes each: roughly 400 MB at the defaults, and every element is initialised before mining starts. Lower `-I` first:

```bash
1miner profanity -z <pubkey> --leading 0 -I 4096
```

That reduces both memory and start-up time, at some throughput cost.

## "1nft requires --mint-for"

The 1inch Address NFT salt embeds the account being minted for, so the address is meaningless without it. Similarly `--deployer` is required in every salt mode: it is never assumed, because a salt mined against the wrong deployer gives an address that looks valid and is unusable. Both may be omitted only under `--benchmark`, where the output is a rate rather than a result.

## Speeds look lower than published numbers

- **Thermal drift.** Sustained runs settle well below the first few seconds. See [benchmarking.md](benchmarking.md).
- **Comparing modes.** create3 does two keccaks per candidate and lands near half of create2. That is the algorithm, not the implementation.
- **The wrong backend on macOS.** Try `--backend metal` against `--backend opencl` and measure; do not assume.

## The container ignores my subcommand

Fixed, but worth knowing what it looked like: the entrypoint treats an argument that is neither a known subcommand nor a flag as a plain command to execute, so that `docker run ... clinfo` works. An unrecognised subcommand therefore fails with `exec: <name>: not found` rather than a clap error. Check the spelling against `docker run --rm 1miner --help`.

## Suspect the CPU backend's SIMD path

On aarch64 the salt modes use a two-lane NEON Keccak. Both paths are asserted to produce identical addresses, but if you suspect it:

```bash
MINER_NO_NEON=1 1miner self-test --backend cpu
MINER_NO_NEON=1 1miner create3 --deployer 0x... --leading 0 --backend cpu
```

If the scalar path passes and NEON does not, that is a bug worth reporting with your CPU model.

## Metal is slower than expected

If you have modified `salt.metal`, check for register spilling first. The original implementation used a scratch array with computed indices and ran at half speed because the array lived in memory; literal indices keep all 25 keccak lanes in registers.
