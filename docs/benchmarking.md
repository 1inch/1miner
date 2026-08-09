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
- **Excludes a warmup from the figure.** `-w`, default 10 seconds, is passed to the miner as `--warmup`, and the script reads the `Measured:` line the miner prints when it stops. That line is the average over everything after the warmup, so kernel compilation and the first slow round are genuinely left out rather than merely diluted.
- **Alternates the order each pass.** Being second is a real penalty, so it must not always land on the same contender. If your means still depend on order, drift is dominating and the numbers are not yet meaningful.
- **Repeats.** `-p`, default two passes, and it prints the min and max alongside the mean so you can see the spread rather than trusting a single figure.
- **Warns if** `self-test` **fails** on the first backend, because a fast wrong kernel is the failure this project is arranged to prevent.
- **Records provenance** with `-o`: date, host, mode, backend and kernel, since an unlabelled hashrate is not reproducible across driver releases.

## Two numbers, and which one to quote

While mining, the live line is a **rolling window** over the last few seconds. It settles within about a second and then tracks the current rate, so it is the number that shows a GPU throttling, and it reads zero if a device stops producing work.

When the run ends, the miner prints one more line:

```
Measured: 356.101 MH/s over 20.0s (7147094528 hashes)
```

That is the average over everything after `--warmup`, and it is the figure to quote. Quoting the live line instead means quoting a short window, which moves around.

Earlier versions of this project reported a cumulative average since the start of the run, which is worth knowing about because it was wrong in a specific direction: it crept upwards for tens of seconds as it slowly forgot the slow first round, understated every rate, and could never show throttling at all. Figures measured that way were low by a few percent for the salt modes and by about 12% for profanity, whose start-up initialises millions of points before any hashing begins.

A healthy result looks like this — note that reversing the order in pass 2 moved nothing much:

```
pass 1
  opencl:tuned     358.507 MH/s
  metal:tuned      347.959 MH/s
pass 2
  metal:tuned      356.421 MH/s
  opencl:tuned     353.695 MH/s

mean per contender:
  opencl:tuned     356.101 MH/s (min 353.695, max 358.507)
  metal:tuned      352.190 MH/s (min 347.959, max 356.421)
```

For reference, the Rust-versus-C++gate produced 702.1 and 672.9 MH/s for the C++ binary against 712.7 and 696.4 for Rust on the identical kernel — a consistent downward drift across all four runs, with the two hosts otherwise indistinguishable.

## Comparing kernels

```bash
scripts/bench.sh -k "tuned plain" -p 3
```

Check agreement before you believe a speedup: `cargo test --test derivation` and `1miner self-test --kernel <name>`. A kernel that is fast and wrong is worse than no kernel.

## Recording results

Note the GPU, the driver or OS version, the backend, the kernel variant, the mode and the date. Hashrates move with driver releases, so an unlabelled number is not reproducible.

Measured on an Apple M4 Max (40-core GPU), macOS 26.5, `scripts/bench.sh` with an 8-second warmup and a 20-second measured window, two passes in alternating order, August 2026. Each figure is the mean, with the spread in brackets:

| Mode      | Backend     | Kernel   | Speed                                        |
| --------- | ----------- | -------- | -------------------------------------------- |
| create2   | Metal       | built-in | 724.4 MH/s (720.6–728.1)                     |
| create2   | OpenCL      | tuned    | 721.8 MH/s (715.9–727.6)                     |
| profanity | OpenCL      | tuned    | 381.6 MH/s (378.8–384.5)                     |
| create3   | OpenCL      | tuned    | 356.1 MH/s (353.7–358.5)                     |
| create3   | Metal       | built-in | 352.2 MH/s (348.0–356.4)                     |
| create3   | OpenCL      | plain    | 330.0 MH/s (324.8–335.2)                     |
| create3   | CPU, NEON   | built-in | 87.3 MH/s (86.7–88.0, 16 threads)            |
| create3   | CPU, scalar | built-in | 36.2 MH/s (36.1–36.2, `MINER_NO_NEON=1`)     |

For comparison, the C++ references on the same machine: ERADICATE2 at about 733 MH/s for create2, ERADICATE3 at 357 MH/s for create3, profanity2 at 353 MH/s. Those use a rolling window, so they were already honest figures.

Three things the table says:

**The tuned Keccak is worth about 7%** on create3, 356.1 against 330.0. An earlier measurement using the old cumulative averaging put the two within noise of each other and this page said to treat them as equal; that was the measurement's fault, not the kernels'. This is exactly the kind of difference selectable kernels exist to find, so measure on your own hardware rather than trusting either figure.

**Metal and OpenCL are close on this machine**, with OpenCL marginally ahead on create3 and Metal marginally ahead on create2. The spreads overlap in both cases, so neither is a clear winner here. On Apple silicon that is a little surprising given OpenCL is deprecated, and it means the choice is worth measuring rather than assuming.

**The two-lane NEON Keccak is worth 2.4x** on the CPU path, 87.3 against 36.2, for identical output. Still far below the same machine's GPU, which is why the CPU backend is a reference and a fallback rather than a contender.