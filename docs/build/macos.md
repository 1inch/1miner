# Building on macOS

Apple silicon and Intel both work.

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

cargo build --release --features metal
./target/release/1miner self-test --backend metal
```

`--features metal` is what enables the Metal backend. Without it you get an OpenCL-only build that still runs, just without `--backend metal`.

## Which backend to use here

Apple deprecated OpenCL. It still works and still sees the GPU, but it is capped at OpenCL 1.2 and is not the path the driver is tuned for. Metal is the native one.

In practice, on an M4 Max the two are close, with Metal slightly ahead for create3 (362 against 359 MH/s). Measure on your own machine rather than assuming either way:

```bash
1miner create3 --deployer 0x0000000000000000000000000000000000000000 \
  --benchmark --backend opencl --seconds 30
sleep 30
1miner create3 --deployer 0x0000000000000000000000000000000000000000 \
  --benchmark --backend metal --seconds 30
```

`profanity` needs OpenCL either way: it requires secp256k1 arithmetic on the GPU and only the OpenCL kernel provides it.

## Sandboxing

Some sandboxes and remote shells block GPU access, and OpenCL then reports zero devices on a machine that plainly has a GPU. If `1miner` finds nothing but the Displays section of System Information shows your GPU, try running from a normal terminal before investigating anything else.

## Xcode command line tools

Metal shaders are compiled at run time through the system Metal framework, so no extra toolchain is required beyond the command line tools.
