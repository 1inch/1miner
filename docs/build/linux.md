# Building on Linux

Works on x86_64 and aarch64.

```bash
sudo apt-get install -y build-essential opencl-headers ocl-icd-opencl-dev clinfo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

cargo build --release
./target/release/1miner self-test
```

The binary lands at `target/release/1miner`. Kernels are embedded, so it can be copied anywhere on its own.

## OpenCL runtime

Building needs the headers and the ICD loader. *Running* needs your vendor's runtime, which is separate:

| Vendor | Package |
| --- | --- |
| NVIDIA | the proprietary driver, which ships `libnvidia-opencl.so.1` |
| AMD | ROCm OpenCL, or `mesa-opencl-icd` for the open stack |
| Intel | `intel-opencl-icd` |

Confirm the device is visible before anything else:

```bash
clinfo -l
```

If that lists nothing, 1miner will not find a GPU either. See [troubleshooting.md](../troubleshooting.md).

## Static-ish builds

The binary links the OpenCL ICD loader dynamically, which is unavoidable: the loader is what finds the vendor driver at run time. To move a build between machines, either match the distro or use the Docker image, which pins the runtime alongside the binary.

## CPU only

No OpenCL at all is fine if you only want the fallback:

```bash
cargo build --release --no-default-features
./target/release/1miner self-test --backend cpu
```
