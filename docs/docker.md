# Docker

A prebuilt image is published to GHCR from [`.github/workflows/docker.yml`](../.github/workflows/docker.yml) on every push to `main` and on `v*` tags:

```bash
docker run --rm --gpus all ghcr.io/1inch/1miner:latest self-test
docker run --rm --gpus all ghcr.io/1inch/1miner:latest create3 --deployer 0xFactory --leading 0
```

To build locally:

```bash
docker build -t 1miner .
docker run --rm --gpus all 1miner self-test
```

The image is a multi-stage build: Rust on Debian bookworm to compile, then a `debian:bookworm-slim` runtime carrying only the ICD loader, `clinfo` and the binary. It comes out around 150 MB.

Kernels are embedded in the binary via `include_str!`, so unlike profanity2 there are no `.cl` files to keep beside it and no working-directory requirement.

## GPU access

`--gpus all` requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) on the host.

The toolkit mounts `libnvidia-opencl.so.1` into the container but does not register it with the ICD loader, so the image writes the vendor file itself:

```
/etc/OpenCL/vendors/nvidia.icd  ->  libnvidia-opencl.so.1
```

Without that, OpenCL reports zero devices inside a container that plainly has a GPU attached. It is the single most common Docker GPU failure for OpenCL workloads.

Check what the runtime actually sees:

```bash
docker run --rm --gpus all 1miner clinfo -l
docker run --rm --gpus all 1miner nvidia-smi
```

Any argument that is not a 1miner subcommand or flag is run as a plain command, which is what makes those work.

## Environment variables

| Variable | Effect |
| --- | --- |
| `MINER_ARGS` | Used as the argument list when none are given on the command line. |
| `MINER_OUTPUT` | Tee all output to this path as well as stdout. |
| `MINER_SKIP_GPU_CHECK` | Set to `1` to skip the startup "no OpenCL devices" warning. |

`MINER_ARGS` earns its place on hosting panels that give you an image and an environment but no easy way to set a command:

```bash
docker run --rm --gpus all \
  -e MINER_ARGS="create3 --deployer 0xFactory --leading 0" \
  -e MINER_OUTPUT=/workspace/run.log \
  -v "$PWD:/workspace" \
  1miner
```

## Keeping results

The working directory is `/workspace`. Mount something there and set `MINER_OUTPUT` so hits survive the container:

```bash
docker run --rm --gpus all -v "$PWD/out:/workspace" \
  -e MINER_OUTPUT=/workspace/hits.log \
  1miner create3 --deployer 0xFactory --leading 0
```

## Building for another architecture

A build on an Apple silicon machine produces an arm64 image, which will not run on the x86_64 hosts most GPU rentals provide. Cross-build explicitly:

```bash
docker buildx build --platform linux/amd64 -t 1miner:amd64 --load .
```

## Multi-GPU

All visible devices are used, one thread each. Restrict with either Docker or 1miner:

```bash
docker run --rm --gpus '"device=0,1"' 1miner create3 --deployer 0x... --leading 0
docker run --rm --gpus all 1miner create3 --deployer 0x... --leading 0 --skip 2 --skip 3
```
