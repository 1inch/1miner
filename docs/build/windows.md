# Building on Windows

```powershell
# Rust with the MSVC toolchain
winget install Rustlang.Rustup
rustup default stable-x86_64-pc-windows-msvc

cargo build --release
.\target\release\1miner.exe self-test
```

## OpenCL

Windows GPU drivers install `OpenCL.dll` into the system directory, so a machine with a working NVIDIA, AMD or Intel driver already has the runtime.

For linking you need an import library. The most reliable route is the vendor SDK:

- NVIDIA: the CUDA Toolkit ships OpenCL headers and `OpenCL.lib`.
- AMD: the ROCm or legacy APP SDK.
- Intel: the oneAPI Base Toolkit.

If the build cannot find `OpenCL.lib`, point the linker at it:

```powershell
$env:LIB = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;$env:LIB"
cargo build --release
```

Check the runtime side with a `clinfo` build for Windows, or just run `1miner self-test` and read the device list it prints.

## MSYS2 / MinGW

A GNU-toolchain build works too:

```bash
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-opencl-headers mingw-w64-x86_64-opencl-icd
rustup default stable-x86_64-pc-windows-gnu
cargo build --release
```

## CPU only

If you do not want to deal with an OpenCL SDK at all:

```powershell
cargo build --release --no-default-features
.\target\release\1miner.exe self-test --backend cpu
```

## Note on support

Linux and macOS are the platforms this project is exercised on; Windows is expected to work but is less travelled. The Docker image is often the shorter path on a Windows host with an NVIDIA GPU, via WSL2 and the container toolkit.
