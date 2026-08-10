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

- **Cools down before every run.** `-c`, default 60 seconds idle.
- **Throws away whole passes first.** `-x`, default one. Their figures are printed and then ignored. This is not the same thing as `-w`; see [the cooldown is not the whole problem](#the-cooldown-is-not-the-whole-problem) below.
- **Excludes a warmup from the figure.** `-w`, default 10 seconds, is passed to the miner as `--warmup`, and the script reads the `Measured:` line the miner prints when it stops. That line is the average over everything after the warmup, so kernel compilation and the first slow round are genuinely left out rather than merely diluted.
- **Alternates the order each pass.** Being second is a real penalty, so it must not always land on the same contender. If your means still depend on order, drift is dominating and the numbers are not yet meaningful.
- **Repeats.** `-p`, default two passes, and it prints the min and max alongside the mean so you can see the spread rather than trusting a single figure.
- **Says when that spread is too wide to mean anything.** If a contender's passes differ by more than 3% of its mean it prints a warning naming the contender and the percentage, because a mean over runs that disagree by 20% is otherwise reported in exactly the shape of a result.
- **Warns if** `self-test` **fails** on the first backend, because a fast wrong kernel is the failure this project is arranged to prevent.
- **Names the binary it is about to time**, and warns when anything under `crates/` or `kernels/` is newer than it. The candidate list prefers `./target/release/1miner`, which is the previous build until you rebuild, and benchmarking a change you have not compiled reports the change as free. `MINER=/path/to/1miner` overrides the choice.
- **Says why a run produced no rate** instead of only that it did. A backend that refuses the mode, a flag a older binary does not know, a device already busy: all of them used to arrive as the single word FAILED, with the explanation dropped by the same pipe that reads the rate.
- **Collapses kernel variants for backends that have one.** `-k` selects an OpenCL keccak source; Metal and CPU ignore it, so they appear once, as `metal:builtin` rather than as two identical contenders.
- **Records provenance** with `-o`: date, host, mode, backend, kernel, the min and max, and the flags that produced the row. Neither an unlabelled hashrate nor one whose procedure went unrecorded is reproducible.

## The cooldown is not the whole problem

Both of those defaults were raised after measuring what the previous ones did, and `-x` exists because cooling down between runs turned out not to be sufficient on its own.

Holding everything else fixed and putting a 30-second cooldown against 60 on create3, three local work sizes as the contenders, three passes each with every contender rotated through every position:

| cooldown | mean over nine runs | widest spread within one contender |
| --- | --- | --- |
| 30s | 358.06 MH/s | 15.7 MH/s |
| 60s | 363.12 MH/s | 7.2 MH/s |

Thirty seconds costs 1.4% and roughly triples the spread, which is enough to invent a winner: in the 30-second set the contender that happened to draw the first and coldest slot came out 1% ahead, and at 60 seconds all three finished within 0.1% of one another. Local work size makes no measurable difference here at all — a conclusion only the longer cooldown was able to reach.

A cooldown also cannot undo the state the machine was already in when the benchmark started. Run at the old defaults directly after a heavy GPU session, four passes of a single unchanged configuration gave:

```
pass 1  opencl:tuned     284.543 MH/s
pass 2  opencl:tuned     317.898 MH/s
pass 3  opencl:tuned     346.768 MH/s
pass 4  opencl:tuned     341.876 MH/s
```

A 22% spread, a reported mean of 322.771 sitting 11% below the settled figure, and — the part that breaks the harness's own reasoning — a drift that climbs rather than falls, because the machine was recovering rather than heating. Alternating order cancels a drift that runs one way for the whole run; against this it does nothing. Discarded passes do: they absorb the recovery, and printing their figures is what tells you whether the cooldown was long enough, because a discarded pass far below the measured ones means it was not.

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

## What turned out not to matter

Two tuning flags repay far less attention than their presence suggests, at least on an M4 Max against the OpenCL backend. Both are cheap to sweep, and worth sweeping before optimising anything that sits behind them.

`--work` makes no measurable difference. On create3, `0`, `32` and `128` land within 0.1% of one another once the cooldown is long enough for that to be visible. On profanity, `32`, `64` and `128` tie, with `0` and `256` about 3% behind.

Lowering `--inverse-size` only ever loses. Holding the round size fixed at its default 4177920 points and moving the split between `-i` and `-I`:

| `-i` × `-I` | private memory per work item | mean | against the default |
| --- | --- | --- | --- |
| 255 × 16384 | 16320 B | 354.3 MH/s | — |
| 128 × 32640 | 8192 B | 350.5 MH/s | −1.1% |
| 64 × 65280 | 4096 B | 340.0 MH/s | −4.1% |
| 32 × 130560 | 2048 B | 306.7 MH/s | −13.4% |
| 16 × 261120 | 1024 B | 246.2 MH/s | −30.5% |

The extra modular inversions outweigh whatever the smaller private arrays win back, so `-i` is an escape valve for memory pressure rather than a speed knob. That also bounds how much those arrays cost in the first place: halving them moved the figure by one percent, in the direction the extra inversions alone predict. On a GPU where private memory means scratch rather than a slice of unified memory the picture may differ, which is the reason to re-run the sweep there rather than carry this conclusion over.

The private-memory column is an OpenCL fact. The Metal kernel keeps the same prefix products in the buffer it writes anyway, so `-i` there buys only the inversion trade and the middle column does not apply.

`-I` does matter on Metal, and for a reason that has nothing to do with the arithmetic. Each round costs a fixed amount on top of its candidates, so smaller rounds pay it more often: 154 MH/s at `-I 1024`, 335 at 4096, about 397 at the default 16384. Part of that was the host holding the GPU idle while it read the round, which is why the profanity loop now keeps one round in flight; see below. The rest is dispatch overhead on the device and does not go away.

These figures were taken at a short cooldown, so trust the ranking rather than the rates: it is monotone, it survived reversing the order, and the default won from the hottest slot.

## Recording results

Note the GPU, the driver or OS version, the backend, the kernel variant, the mode and the date. Hashrates move with driver releases, so an unlabelled number is not reproducible.

Measured on an Apple M4 Max (40-core GPU), macOS 26.5, `scripts/bench.sh` with an 8-second warmup and a 20-second measured window, two passes in alternating order, August 2026. Each figure is the mean, with the spread in brackets.

These were taken before the 60-second cooldown and the discarded pass, so treat them as a set that is internally consistent but around one to two percent low, with spreads narrower than that procedure can actually support; the whole table is due a re-measurement under the current defaults. The one row re-measured since is create3 on OpenCL, which came out at 363.3 MH/s (358.8–366.1) over three rotated passes at a 60-second cooldown, against the 356.1 below:

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

### profanity on Metal

A discarded pass then two measured passes in rotating order, 40-second cooldowns, 8-second warmup, 20-second measured window, August 2026:

| Backend | discarded | pass 1 | pass 2 |
| --- | --- | --- | --- |
| Metal | 387.8 | 321.5 | 407.6 |
| OpenCL | 362.5 | 325.0 | 356.1 |

Pass 1 is low for both, which is the useful part of the result: whatever slowed it was not a property of either backend, and reporting the mean of the three would have buried that in a single number. Excluding it, Metal runs about 397 MH/s against OpenCL's 359 on this machine.

That still puts the two closer than the spread of a single pass, so treat the ordering as suggestive rather than settled. An earlier attempt at the same comparison produced Metal figures ranging from 240 to 401 MH/s, all of the low ones from runs launched immediately after an OpenCL process — which is the failure mode alternating and discarding a pass exists to expose, and the reason a single ordering is worth nothing here.

### Measuring a change rather than a backend

The same noise that makes 397-against-359 shaky makes a 2% change invisible, so measure a change where its effect is largest rather than where users will meet it. Pipelining the Metal round loop moves a fixed per-round cost, so shrinking the round amplifies it:

| `-I` | one round at a time | one round in flight | change |
| --- | --- | --- | --- |
| 1024 | 135.6, 136.5 | 154.4, 154.5 | +13.5% |
| 4096 | 312.9 | 334.6 | +6.9% |

Both repeats land within 1% of each other, against a default-tuning comparison of the same two binaries that could not separate them at all. Converted to time per round the two rows agree on what was actually saved — 0.235 ms and 0.217 ms — which is the check that the model behind the experiment was right, and it prices the change at about 2% for a default run.

Three things the table says:

**The tuned Keccak is worth about 7%** on create3, 356.1 against 330.0. An earlier measurement using the old cumulative averaging put the two within noise of each other and this page said to treat them as equal; that was the measurement's fault, not the kernels'. This is exactly the kind of difference selectable kernels exist to find, so measure on your own hardware rather than trusting either figure.

**Metal and OpenCL are close on this machine**, with OpenCL marginally ahead on create3 and Metal marginally ahead on create2. The spreads overlap in both cases, so neither is a clear winner here. On Apple silicon that is a little surprising given OpenCL is deprecated, and it means the choice is worth measuring rather than assuming.

**The two-lane NEON Keccak is worth 2.4x** on the CPU path, 87.3 against 36.2, for identical output. Still far below the same machine's GPU, which is why the CPU backend is a reference and a fallback rather than a contender.