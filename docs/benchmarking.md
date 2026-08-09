# Benchmarking

```bash
1miner create3 --deployer 0x0000000000000000000000000000000000000000 \
  --benchmark --seconds 30
```

`--benchmark` scores nothing and reports throughput only. It is the one place where `--deployer` may be omitted, because the output is a rate rather than a usable address.

## Rates are not comparable across modes

- `create2` runs **one** keccak per candidate.
- `create3` and `1nft` run **two**: CREATE2 for the proxy, then CREATE for the contract.
- `profanity` does secp256k1 point arithmetic and a keccak.

So create3 lands at roughly half of create2 on the same GPU, and that ratio is a property of the algorithm, not of the implementation. Only compare like with like.

## Comparing fairly

GPUs throttle. A back-to-back A/B where one contender runs on a GPU the other just heated for 20 seconds will show a difference that is entirely thermal. When this project chose Rust over C++, the first naive measurement showed Rust 22% ahead purely because it ran first.

`scripts/bench.sh` encodes the discipline that fixes this, so you do not have to remember it:

```bash
scripts/bench.sh -b "opencl metal"        # backends, one mode
scripts/bench.sh -k "tuned plain" -p 3    # kernel variants, three passes
scripts/bench.sh -m create2 -b "opencl metal" -o bench-results.md
```

What it does, and what to reproduce if you measure by hand:

- **Cools down before every run.** `-c`, default 30 seconds idle.
- **Warms up inside every run.** `-w`, default 10 seconds discarded, then `-d` seconds measured, so start-up and kernel compilation are excluded.
- **Alternates the order each pass.** Being second is a real penalty, so it must not always land on the same contender. If your means still depend on order, drift is dominating and the numbers are not yet meaningful.
- **Repeats.** `-p`, default two passes, and it prints the min and max alongside the mean so you can see the spread rather than trusting a single figure.
- **Warns if `self-test` fails** on the first backend, because a fast wrong kernel is the failure this project is arranged to prevent.
- **Records provenance** with `-o`: date, host, mode, backend and kernel, since an unlabelled hashrate is not reproducible across driver releases.

A healthy result looks like this — note that reversing the order in pass 2 moved nothing:

```
pass 1
  opencl:tuned     352.864 MH/s
  metal:tuned      367.690 MH/s
pass 2
  metal:tuned      367.317 MH/s
  opencl:tuned     353.847 MH/s

mean per contender:
  opencl:tuned     353.356 MH/s (min 352.864, max 353.847)
  metal:tuned      367.504 MH/s (min 367.317, max 367.690)
```

For reference, the Rust-versus-C++ gate produced 702.1 and 672.9 MH/s for the C++ binary against 712.7 and 696.4 for Rust on the identical kernel — a consistent downward drift across all four runs, with the two hosts otherwise indistinguishable.

## Comparing kernels

```bash
scripts/bench.sh -k "tuned plain" -p 3
```

Check agreement before you believe a speedup: `cargo test --test derivation` and `1miner self-test --kernel <name>`. A kernel that is fast and wrong is worse than no kernel.

## Recording results

Note the GPU, the driver or OS version, the backend, the kernel variant, the mode and the date. Hashrates move with driver releases, so an unlabelled number is not reproducible.

Measured on an Apple M4 Max (40-core GPU), macOS 26.5, 20-second windows after a 10-second warmup, August 2026:

| Mode | Backend | Kernel | Speed |
| --- | --- | --- | --- |
| create2 | OpenCL | tuned | 722.6 MH/s |
| create2 | Metal | built-in | 728.9 MH/s |
| create3 | OpenCL | tuned | 358.6 MH/s |
| create3 | OpenCL | plain | 346.6 MH/s |
| create3 | Metal | built-in | 362.4 MH/s |
| profanity | OpenCL | tuned | 338.9 MH/s (`-I 8192`) |
| create3 | CPU, NEON | built-in | 88.7 MH/s (16 threads) |
| create3 | CPU, scalar | built-in | 38.0 MH/s (`MINER_NO_NEON=1`) |

For comparison, the C++ references on the same machine: ERADICATE2 at about 733 MH/s for create2, ERADICATE3 at 357 MH/s for create3, profanity2 at 353 MH/s.

The tuned and plain OpenCL figures above were taken in the same session and are within the run-to-run spread, so treat them as equal on this hardware rather than as a 3% win.

The CPU rows show what the two-lane NEON Keccak buys on aarch64: 2.3x, for identical output. It is still far below the same machine's GPU, which is why the CPU backend is a reference and a fallback rather than a contender.

One finding worth repeating: the first Metal kernel ran at 193 MH/s because its permutation used a scratch array with computed indices, which spilled to memory. Rewriting it with literal indices so all 25 lanes stay in registers took it to 362 MH/s for no change in output. If a kernel is unexpectedly slow, look at register spilling before anything else.
