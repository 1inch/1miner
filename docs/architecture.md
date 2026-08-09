# Architecture

```
crates/
  miner-core/      address maths, scoring, mode configuration. No GPU, no I/O.
  miner-backend/   the Backend trait and its OpenCL, Metal and CPU implementations,
                   plus the two-lane NEON Keccak used by the CPU path on aarch64.
  miner-cli/       clap surface, output, self-test.
kernels/
  opencl/          keccak_tuned.cl, keccak_plain.cl, salt.cl, profanity.cl
  metal/           salt.metal
```

## Two engine families

The four modes split into two shapes, and the split runs deeper than the CLI suggests:

- **Salt search** (`create2`, `create3`, `1nft`) iterates a salt through keccak.
- **Key search** (`profanity`) iterates a private-key offset through secp256k1 point addition.

They share keccak, address scoring, result handling, the CLI and the Docker image. They differ only in the GPU inner loop.

## The three salt modes are one kernel

ERADICATE2 and ERADICATE3, the upstream miners for CREATE2 and CREATE3, turn out to be the same program with one difference: CREATE3 runs a second keccak. Their `keccak.cl` files are byte-identical and so are all eight scoring functions.

So `kernels/opencl/salt.cl` covers all three modes:

- `SALT_SECOND_HASH` is defined for create3 and 1nft and absent for create2.
- `SALT_INITHASH` carries the 85-byte pre-image, which differs by what goes in the code-hash field.
- 1nft is create3 with the low 16 salt bytes pinned before the run starts, which is a host-side decision needing no kernel support at all.

## Constants: compile time or run time

The OpenCL kernel takes the pre-image as a `-D` define, following the reference implementations, which lets the compiler fold the first keccak rounds. The cost is that changing deployer, code hash or base salt forces a rebuild. Compiled binaries are cached under `$XDG_CACHE_HOME/1miner/opencl` keyed on the source, the options and the device name, so the rebuild is paid once.

The Metal path passes the state in a buffer instead and never rebuilds. That is worth knowing if you are considering the same for OpenCL: measure first, since the constant is likely what makes the OpenCL kernel as fast as it is.

## Work-item coordinates

Only three 32-bit words of the salt vary:

```
h.d[6] += deviceIndex
h.d[7] += get_global_id(0)
h.d[8] += round
```

`SaltConfig::salt_at` reproduces this on the host. That matters because the kernel *reconstructs* the salt in its result path rather than carrying it through the hash, so the reconstruction is separate code that can drift from the hashing path. Re-deriving on the host is what catches it.

## Backends

```rust
pub trait Backend {
    fn name(&self) -> &'static str;
    fn devices(&self) -> &[DeviceInfo];
    fn run(&mut self, job: &Job, reporter: &mut dyn Reporter,
           should_stop: &(dyn Fn() -> bool + Sync)) -> Result<()>;
}
```

A `Job` bundles the mode configuration, the scoring specification, the kernel choice and the tuning. Backends report through `Reporter`, which the CLI implements as a live speed line plus a printed line per improved hit.

OpenCL runs one thread per device, each with its own context and queue, sharing a best-score atomic so a strong hit on one GPU raises the bar on all of them. The per-device loop mirrors the reference dispatcher: enqueue a non-blocking read of the previous round's results, queue the next round behind it, and wait only on the read, which keeps a kernel in flight at all times.

Metal runs single threaded, since there is one system default device.

## Where to add things

- **A mode** — extend `MineMode` and `SaltConfig` in `miner-core`, add a subcommand in `miner-cli/src/cli.rs`. If it is salt-shaped it likely needs no kernel change.
- **A backend** — implement `Backend`, add a `--backend` value, extend `open_backend` in `miner-cli/src/main.rs`. CUDA would slot in here.
- **A kernel variant** — see [backends.md](backends.md).

Whatever you add, make it pass `cargo test --test derivation` and `1miner self-test` before trusting a benchmark from it.

## Documentation conventions

Paragraphs are written as a single line each, with no manual wrapping. Let the editor or renderer wrap them: hard breaks at a fixed column wrap badly in Markdown viewers that have their own width, and they turn a one-word edit into a diff across several lines.

Line breaks belong only where Markdown needs them, which means between blocks, and inside lists, tables and code blocks.
