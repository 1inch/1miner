# 1miner

One GPU miner for four kinds of Ethereum vanity address, with swappable compute backends and kernels.

| Mode | What it searches | What you get |
| --- | --- | --- |
| `profanity` | private-key offsets from a seed **public** key | an offset to add to your seed private key |
| `create2` | CREATE2 salts for a deployer and init code | a `bytes32` salt |
| `create3` | CREATE3 salts for a factory | a `bytes32` salt |
| `1nft` | 1inch Address NFT magics | a `bytes16` magic for `mint()` / `mintFor()` |

Backends are chosen at run time: **OpenCL** by default and the only one that covers every mode, **Metal** on macOS, and a portable **CPU** fallback. CUDA is the intended next addition.

## Quick start

```bash
cargo build --release                    # OpenCL
cargo build --release --features metal   # plus the macOS Metal backend

# Confirm the GPU agrees with the CPU reference before trusting a long run.
1miner self-test

# A CREATE3 address starting with as many zeros as possible.
1miner create3 --deployer 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf --leading 0

# A CREATE2 address matching a prefix.
1miner create2 --deployer 0xYourFactory --init-code 0x60806040... --matching dead

# A 1inch Address NFT magic for one account.
1miner 1nft --deployer 0xDeployer --mint-for 0xYourAccount --zero-bytes

# A vanity account address, without ever handing over a private key.
1miner profanity --public-key <128 hex chars> --leading 0

# Report every address matching a mask in full, instead of climbing to a best score.
1miner create3 --deployer 0xFactory --exact deadbeef
```

Every mode guide ends with how to turn the result into a deployed contract or a usable key. Start with [docs/modes/](docs/modes/).

## Safety

`profanity` mode never accepts a private key. You generate a keypair offline, pass only the public key, and the miner reports an **offset**. Adding that offset to your seed private key gives the private key for the found address, so the search itself can run on a machine you do not control. See [docs/modes/profanity.md](docs/modes/profanity.md).

Every reported hit is re-derived on the CPU before it is printed. If the kernel and the reference ever disagree, the hit is flagged and the process exits non-zero rather than handing you an address that does not exist. Disable with `--no-verify` only if you have a reason to.

No mode ever assumes a deployer address. A salt mined against the wrong deployer produces an address that looks perfectly valid and is unusable, so `--deployer` is always required outside `--benchmark`.

## Performance

Measured on an Apple M4 Max (40-core GPU), 20-second windows after warmup:

| Mode | Backend | Speed |
| --- | --- | --- |
| create2 | OpenCL | 722.6 MH/s |
| create2 | Metal | 728.9 MH/s |
| create3 / 1nft | OpenCL | 358.6 MH/s |
| create3 / 1nft | Metal | 362.4 MH/s |
| profanity | OpenCL | 338.9 MH/s |
| create3 / 1nft | CPU (NEON, 16 threads) | 88.7 MH/s |

CREATE3 runs at about half of CREATE2 because it hashes twice: CREATE2 for the proxy, then CREATE for the contract. Rates are not comparable across modes for that reason. See [docs/benchmarking.md](docs/benchmarking.md).

## Docker

Kernels are embedded in the binary, so the image needs no files beside it.

```bash
docker build -t 1miner .
docker run --rm --gpus all 1miner self-test
docker run --rm --gpus all 1miner create3 --deployer 0xFactory --leading 0
```

For rented GPUs see [docs/vastai.md](docs/vastai.md).

## Documentation

- Modes: [profanity](docs/modes/profanity.md), [create2](docs/modes/create2.md), [create3](docs/modes/create3.md), [1nft](docs/modes/1nft.md)
- [How address derivation works](docs/how-address-derivation-works.md)
- [Backends and kernels](docs/backends.md), [benchmarking](docs/benchmarking.md)
- Building: [Linux](docs/build/linux.md), [macOS](docs/build/macos.md), [Windows](docs/build/windows.md)
- [Docker](docs/docker.md), [vast.ai](docs/vastai.md)
- [Correctness](docs/correctness.md), [architecture](docs/architecture.md), [troubleshooting](docs/troubleshooting.md)

## Credits

The GPU kernels descend from [profanity2](https://github.com/1inch/profanity2) and from ERADICATE2 / ERADICATE3 by Johan Gustafsson. See [LICENSE](LICENSE).

## License

MIT.
