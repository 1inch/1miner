# AGENTS.md

`1miner` is a GPU miner for four kinds of vanity Ethereum address — `profanity`, `create2`, `create3`, `1nft` — with compute backends and Keccak kernels selectable at run time. Rust workspace, edition 2024, MSRV 1.85. The binary is named `1miner`.

## Layout

```
crates/miner-core/     address maths, scoring, mode config. CPU only, no I/O. The reference.
crates/miner-backend/  the Backend trait, its OpenCL/Metal/CPU impls, NEON keccak, speed metering.
crates/miner-cli/      clap surface, terminal output, self-test.
kernels/opencl/        keccak_tuned.cl, keccak_plain.cl, salt.cl, profanity.cl
kernels/metal/         salt.metal
docs/                  user and contributor documentation.
scripts/bench.sh       benchmark harness.
references/            upstream C++ miners (profanity2, ERADICATE2/3). Gitignored, read-only.
```

The three salt modes share one kernel: `create3`/`1nft` add a second keccak, and `1nft` pins the low 16 salt bytes host-side. Kernel sources are embedded with `include_str!` in `crates/miner-backend/src/kernels.rs`, so the binary is self-contained — never add a runtime file lookup for a kernel.

## Commands

```bash
cargo build --release                     # OpenCL, the default feature
cargo build --release --features metal    # plus Metal, macOS only
cargo test                                # add --features metal on macOS
cargo test --test derivation              # cross-backend agreement tests
./target/release/1miner self-test         # check the device actually present
scripts/bench.sh -b "opencl metal"        # see docs/benchmarking.md before quoting a number
```

GPU-dependent tests print a `skipping` line and pass when no device is present, so the suite still runs in CI and in a container. Keep it that way.

## Non-negotiables

Every failure mode here is silent: a wrong pad byte or the wrong nonce still produces a well-formed address, and nothing in the output says so.

- **Never accept a private key.** `profanity` takes a public key and reports an offset. That is what makes running on rented hardware safe. Do not add a private-key flag.
- **Never assume a deployer.** `--deployer` is required outside `--benchmark`; a salt mined against the wrong deployer gives a valid-looking, unusable address.
- **Verification stays on by default.** Every hit is re-derived on the CPU before it is printed; a disagreement prints `[UNVERIFIED: ...]` and the process exits non-zero.
- **Keep the CPU reference independently written.** `miner-core` builds ordinary byte strings and hashes them with the `sha3` crate, deliberately not copying the kernels' split-padding trick. Two implementations that share a shortcut share its bugs.
- **Touch a kernel or backend, run the agreement tests before believing any number.** `cargo test --test derivation` and `1miner self-test --kernel <name>`. Fast and wrong is the outcome this project is arranged to prevent.
- **Scoring functions are line-by-line ports, not idiomatic rewrites.** `ScoreFn` discriminants are the wire format the kernels read, so the order is fixed by the OpenCL side. Do not reorder or tidy `crates/miner-core/src/scoring.rs`.
- **The kernel reconstructs the salt separately from hashing it.** Any change to the work-item coordinates (`h.d[6] += device`, `d[7] += global_id`, `d[8] += round`) has to be mirrored in `SaltConfig::salt_at`; the planted-target tests are what catch drift.

## Where to add things

- **A mode** — extend `MineMode` and `SaltConfig` in `miner-core`, add a subcommand in `crates/miner-cli/src/cli.rs`. A salt-shaped mode usually needs no kernel change.
- **A backend** — implement `Backend`, add a `--backend` value, extend `open_backend` in `crates/miner-cli/src/main.rs`. CUDA slots in here.
- **A kernel variant** — source into `kernels/`, a `const` in `kernels.rs`, a `KeccakVariant` arm wiring `source()`/`as_str()`/`parse()`/`all()`, and the name in the `--kernel` value list. Details in `docs/backends.md`.

## Style

- `thiserror` for library errors (`CoreError`, `BackendError`), `anyhow` in the CLI. Messages are lowercase and say what to do instead of what went wrong.
- Comments explain why: a reference implementation's quirk, a measured result, a trap already fallen into. They never narrate the code. Module-level `//!` docs carry the module's place in the correctness story — keep them current.
- **Do not run `cargo fmt` across the tree.** It is not clean under default rustfmt nor under `use_small_heuristics=Max`, so a blanket format buries the real change in churn. Format what you touch and match the surrounding lines.

## Documentation conventions

- Paragraphs are written as a single line each, with no manual wrapping. Let the editor or renderer wrap them: hard breaks at a fixed column wrap badly in Markdown viewers that have their own width, and they turn a one-word edit into a diff across several lines.
- Line breaks belong only where Markdown needs them, which means between blocks, and inside lists, tables and code blocks.

## Benchmarks

Never quote a hashrate from a single back-to-back run. GPUs throttle, and this project's own Rust-versus-C++ comparison first showed Rust 22% ahead purely because it ran on a cold GPU. Use `scripts/bench.sh`, which cools down before each run, excludes a warmup, alternates the order between passes and prints the spread. Quote the `Measured:` line, not the live rolling window. Rates are not comparable across modes: `create3` and `1nft` hash twice, `create2` once. Record GPU, driver or OS version, backend, kernel, mode and date.

## Environment

| Variable | Effect |
| --- | --- |
| `MINER_NO_NEON=1` | Force the scalar CPU keccak instead of the two-lane NEON path. |
| `MINER_ARGS` | Container only: the argument list used when none is given. |
| `MINER_OUTPUT` | Container only: tee output to a file. |
| `MINER_SKIP_GPU_CHECK=1` | Container only: silence the "no OpenCL devices" warning. |

Compiled OpenCL kernels are cached under `$XDG_CACHE_HOME/1miner/opencl`, keyed on source, build options and device name. `--no-cache` bypasses it; changing deployer, code hash or base salt forces a rebuild, because the pre-image is a `-D` define.

## CI

`.github/workflows/docker.yml` builds the image and smoke-tests it without a GPU on every push and pull request, but publishes to GHCR only when the version the binary reports is not in the registry yet. Bump `workspace.package.version` to release: a commit to `main` on its own no longer publishes anything, and a `v*` tag that disagrees with that version fails the build. No workflow runs `cargo test` yet.
