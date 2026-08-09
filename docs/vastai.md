# Running 1miner on a rented GPU (vast.ai)

The [Dockerfile](../Dockerfile) in the repository root builds an image with the binary, the GPU kernels and the OpenCL runtime glue a container needs. Nothing gets installed on the rented machine: you pick the image, type the 1miner command line into the template and start the instance.

Unlike profanity2, the kernels are compiled into the binary, so there are no `.cl` files to keep beside it and no working-directory requirement.

The image is published as `ghcr.io/1inch/1miner` (see [Building and pushing the image](#building-and-pushing-the-image) if you prefer your own registry). It works on any Docker host with an NVIDIA GPU; vast.ai is just the cheapest way to rent one.

## Why this is safe on someone else's machine

In `profanity` mode 1miner never sees your private key. You pass a *public* key with `--public-key`, and what it prints is an **offset** that is worthless without the seed private key that stays on your machine; the two are added together locally afterwards. See [modes/profanity.md](modes/profanity.md). That is what makes renting GPU time from strangers acceptable.

The salt modes never involve a key at all. The worst a hostile host could do is report a salt that does not produce the address it claims, and `--verify` (on by default) re-derives every hit on the CPU before printing it, so that fails loudly rather than silently.

## Quick start on vast.ai

1. Create a template ([cloud.vast.ai/templates](https://cloud.vast.ai/templates/) → *New*):

   | Field | Value |
   |---|---|
   | Image Path:Tag | `ghcr.io/1inch/1miner:latest` |
   | Launch Mode | **docker ENTRYPOINT** |
   | Arguments | `self-test` |
   | Disk Space | 12 GB (the image is about 150 MB, this is just the minimum) |

   Launch mode matters. In the SSH and Jupyter modes vast.ai replaces the image entrypoint with its own setup script, so the miner never starts. ENTRYPOINT mode runs the image as is and appends the *Arguments* field to the entrypoint, which is exactly the 1miner command line.

2. Pick an offer on the [search page](https://cloud.vast.ai/create/) and rent it. 1miner is pure compute, so sort by price.

3. **Read the log before doing anything else.** `self-test` takes about a second and tells you whether the rental is sound:

   ```
   Constants
     ok    proxy bytecode hash

   Known-answer vectors (CPU)
     ok    create3 zero-salt vector

   Backend agreement (opencl)
     ok    create2 planted target
     ok    create3 planted target
     ok    1nft planted target

   All checks passed.
   ```

   If any line says `FAIL`, destroy the instance and rent another. A bad driver or a miscompiled kernel produces plausible-looking wrong addresses, and nothing else will tell you. This is the cheapest minute you will spend on the rental.

4. Edit the template's *Arguments* to the real search and restart the instance. Restarting keeps the same machine, which is the point: the check you just passed applies to the GPU you are about to mine on.

   ```
   create3 --deployer 0xYourFactory --leading 0
   ```

   To skip the restart, set *Arguments* to both from the start. A failed check then exits before any mining happens:

   ```
   sh -c '1miner self-test && 1miner create3 --deployer 0xYourFactory --leading 0'
   ```

5. Watch the instance log (the log button on the instance card, or `vastai logs <instance_id>`). Every improvement is printed as it is found:

   ```
   Mode: create3 via opencl
   Kernel: keccak=tuned
   Devices:
     GPU0: NVIDIA GeForce RTX 4090, 25757220864 bytes available, 128 compute units
   Deployer: 0xyourfactory...

     Time:     6s  Score:  8  GPU0  Salt: 0x8954...2441  Address: 0x00000000834E17ea...
   ```

6. Copy the salt, magic or offset out of the log, then verify it locally before spending anything. Each [mode guide](modes/) ends with how to do that.

Everything on stdout is also appended to `MINER_OUTPUT` inside the instance if you set it, so a result is not lost when the log view scrolls away. In ENTRYPOINT mode the instance log is the only channel out of the container, though — there is no SSH to copy that file over. Save what you need before destroying the instance:

```bash
vastai logs <INSTANCE_ID> --tail 500 > 1miner-results.log
```

If you want a shell as well, rent in the SSH launch mode instead and start the search from the on-start script shown under [Configuration](#configuration).

## The same thing from the CLI

```bash
pip install vastai
vastai set api-key YOUR_API_KEY

# find something cheap with a single 4090
vastai search offers 'gpu_name=RTX_4090 num_gpus=1' -o 'dph' | head

# rent one machine, with self-test gating the real search on that same GPU
vastai create instance <OFFER_ID> \
    --image ghcr.io/YOUR_USER/1miner:latest \
    --disk 12 --label 1miner \
    --env '-e MINER_OUTPUT=/workspace/hits.log' \
    --args sh -c '1miner self-test && 1miner create3 --deployer 0xYourFactory --leading 0'

vastai show instances
vastai logs <INSTANCE_ID>

# no mode stops on its own, so destroy the instance when you have what you wanted
vastai destroy instance <INSTANCE_ID>
```

One instance, not two. Renting a second machine for the real search would defeat the check, because a bad driver or a miscompiled kernel belongs to a particular host: verifying one box and mining on another verifies nothing. An offer ID is consumed by the first rental as well, so reusing it fails and you would need a fresh `search offers`.

Chaining with `&&` means a failed self-test exits the container before any mining starts, and the log ends on the `FAIL` line. You do not pay to mine on a box that cannot derive addresses correctly.

The `sh -c` form works because the entrypoint execs anything that is neither a 1miner subcommand nor a flag. Check that it survives your first run, though: `--args` takes everything after it, and if vast.ai flattens the quoting then `sh -c` gets only the first word as its command, so the container prints the help text and exits with code 2 instead of mining. That is obvious in the log, which is why it is worth one glance after the first launch. If it happens, fall back to the GUI flow above — run `self-test` as the whole job, read the log, then edit *Arguments* and restart the **same** instance, which keeps the same machine.

There is no flag for the launch mode: passing `--args` is what selects it, the way `--ssh` and `--jupyter` select theirs. `--args` must be the **last** option, everything after it goes to the container. Sorting by `dph` lists the cheapest offers first, `dph-` the most expensive.

## Arguments for each mode

Whatever goes in the *Arguments* field or after `--args` is a normal 1miner command line:

```bash
# vanity contract address from a CREATE3 factory
create3 --deployer 0xYourFactory --leading 0

# vanity CREATE2 address, init code read from the argument
create2 --deployer 0xYourFactory --init-code 0x60806040... --matching dead

# 1inch Address NFT magic for one account
1nft --deployer 0xDeployer --mint-for 0xYourAccount --zero-bytes

# vanity account address, public key only
profanity --public-key YOUR_128_HEX_PUBLIC_KEY --leading 0

# every address matching a mask in full, rather than a climbing score
create3 --deployer 0xYourFactory --exact deadbeef

# throughput of this particular card
create3 --deployer 0x0000000000000000000000000000000000000000 --benchmark --seconds 30
```

`--deployer` is required in all three salt modes, and `--mint-for` too in `1nft`. Neither is ever assumed: a salt mined against the wrong deployer gives an address that looks perfectly valid and is unusable. `--benchmark` is the one exception, since its output is a rate rather than a result.

Note that no mode finishes on its own. The salt and profanity modes keep looking for a better score, and `--exact` keeps printing matches until stopped. Either pass `--seconds` for a bounded run or destroy the instance once you have what you wanted — a running instance keeps costing money.

## Configuration

Options can be given as container arguments (the *Arguments* field) or through environment variables (the *Docker Options* field, `-e NAME=value`). Environment variables are the only way to configure the run in the SSH and Jupyter launch modes, where the entrypoint is replaced and the on-start script has to start it:

```bash
env >> /etc/environment
/usr/local/bin/1miner-entrypoint
```

| Variable | Effect |
|---|---|
| `MINER_ARGS` | Arguments to use when none are passed to the container, e.g. `create3 --deployer 0x... --leading 0` |
| `MINER_OUTPUT` | File the output is copied to as well as stdout, e.g. `/workspace/hits.log` |
| `MINER_SKIP_GPU_CHECK` | Set to `1` to silence the startup warning when no OpenCL device is visible |
| `MINER_NO_NEON` | Set to `1` to force the scalar CPU path; only relevant with `--backend cpu` on ARM hosts |

So this template configuration

```
Arguments:      create3 --deployer 0xYourFactory --leading 0
```

and this one

```
Arguments:      (empty)
Docker Options: -e MINER_ARGS="create3 --deployer 0xYourFactory --leading 0"
```

do the same thing.

An argument that is not a 1miner subcommand or a flag is run as a command instead, which is what you want when an instance misbehaves:

```
Arguments: clinfo        # OpenCL devices the container can see
Arguments: nvidia-smi    # driver version and utilisation
Arguments: env           # the whole environment, useful for a name clash
```

The check is an allowlist of subcommands (`profanity`, `create2`, `create3`, `1nft`, `self-test`, `help`) rather than "does it start with a dash", because 1miner's subcommands do not. A misspelled subcommand therefore fails with `exec: <name>: not found` instead of a usage error.

## Choosing an offer

Any NVIDIA card works. Worth checking:

- **Driver version.** The OpenCL runtime comes from the host driver, and very old drivers occasionally fail to expose OpenCL at all. The CUDA version shown on the offer is a reasonable proxy for driver age, even though 1miner uses OpenCL rather than CUDA.
- **Number of GPUs.** All visible devices are used, one thread each, and the score bar is shared between them. `--skip <index>` leaves one out.
- **Memory, for `profanity` only.** It allocates `--inverse-size` times `--inverse-multiple` points across three 32-byte buffers, so roughly 400 MB at the defaults, and initialises all of it before mining starts. Lower `-I` if a card is short on memory or takes too long to start. The salt modes need almost nothing.

## Cost

Runtime grows exponentially with the length of the pattern, so estimate before renting. Each additional leading hex character multiplies expected time by 16: at 350 MH/s, eight leading zeros is minutes and twelve is months. Run the `--benchmark` line above on the instance to get its real rate, then work from that. See [benchmarking.md](benchmarking.md).

## Building and pushing the image

The official image is built and pushed by GitHub Actions (`.github/workflows/docker.yml`) whenever the workspace version in `Cargo.toml` changes, so `:latest` is the last released version rather than the last commit to `main`. Use that unless you have local changes, need a revision that is not published yet, or would rather not depend on a registry someone else controls.

A rented machine can only pull from a registry, so a custom image has to be pushed somewhere first — GHCR if you will use it more than once, ttl.sh for a single throwaway run.

```bash
git clone https://github.com/1inch/1miner
cd 1miner
docker build --platform linux/amd64 -t 1miner .
```

`--platform linux/amd64` is not optional. Rented GPU machines are x86_64, and a native build on an Apple silicon Mac produces an arm64 image that will not run there. Under emulation the build takes a few minutes rather than under one.

### GitHub Container Registry

The image stays until you delete it and the name is readable, which is what you want if you are going to rent machines more than once. Create a personal access token with the `write:packages` scope, then:

```bash
echo YOUR_TOKEN | docker login ghcr.io -u YOUR_USER --password-stdin
docker tag 1miner ghcr.io/YOUR_USER/1miner:latest
docker push ghcr.io/YOUR_USER/1miner:latest
```

A package pushed to GHCR is **private by default**, so a rented machine cannot pull it yet. Either make it public once, under Packages on your GitHub profile (or the org), or hand the credentials to vast.ai:

```bash
# public package
vastai create instance <OFFER_ID> --image ghcr.io/YOUR_USER/1miner:latest \
    --disk 12 --args create3 --deployer 0xYourFactory --leading 0

# private package
vastai create instance <OFFER_ID> --image ghcr.io/YOUR_USER/1miner:latest \
    --login '-u YOUR_USER -p YOUR_TOKEN ghcr.io' \
    --disk 12 --args create3 --deployer 0xYourFactory --leading 0
```

In the GUI the same credentials go into the *Docker login* field of the template, next to the image path.

For the published `ghcr.io/1inch/1miner` image the same applies once the package visibility is Public:

```bash
vastai create instance <OFFER_ID> --image ghcr.io/1inch/1miner:latest \
    --disk 12 --args create3 --deployer 0xYourFactory --leading 0
```

### ttl.sh

[ttl.sh](https://ttl.sh) is an anonymous registry that deletes what you push after the time given in the tag. No account, no login, no cleanup, which suits a one-off search:

```bash
IMAGE=ttl.sh/1miner-$(uuidgen | tr '[:upper:]' '[:lower:]'):24h
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
echo "$IMAGE"
```

The tag is the lifetime and 24 hours is the maximum, so the image needs a unique name instead, hence the UUID. Anyone who learns that name can pull it, which is harmless here: it holds public source code and no key of yours. Keep the lifetime longer than the search, because an instance recreated or moved after the image expired will fail to start.

Whichever registry you use, its full name goes into the *Image Path:Tag* field of the template, or into `--image` on the command line.

## Troubleshooting

**`warning: no OpenCL devices are visible to the container`** — the container has no usable GPU driver. Run `clinfo` in the same image (`Arguments: clinfo`) to see what the ICD loader finds. On a self-hosted machine, check the container was started with `--gpus all` and that the NVIDIA Container Toolkit is installed. On vast.ai, verify the offer really has an NVIDIA GPU; if it does and the warning persists, the host driver is broken — destroy the instance, you should not pay for it.

**`Devices:` is printed but the list is empty** — an OpenCL platform exists but exposes no GPU device, usually a mismatched host driver.

**`self-test` prints `FAIL`** — do not mine on this instance. The GPU is deriving different addresses from the CPU reference, so any result would be wrong in a way that looks right. Destroy it and rent another. If it reproduces across several hosts, that is a bug in 1miner rather than the rental; please report it with the GPU and driver version.

**A hit is printed with `[UNVERIFIED: CPU re-derivation disagrees with the kernel]`** — same situation, caught mid-run. The process exits non-zero on purpose. See [correctness.md](correctness.md).

**The instance stays in `loading` and never runs** — the image could not be pulled. A GHCR package is private until you say otherwise, and a ttl.sh image is gone once the lifetime in its tag has passed. Check `status_msg` in `vastai show instance <id> --raw`.

**The instance shows as exited immediately** — read the log. Without arguments the entrypoint prints the help text and exits. 1miner itself refuses to start when a required input is missing: `--deployer` in any salt mode, `--mint-for` in `1nft`, a valid 128-hex `--public-key` in `profanity`, or any scoring flag at all. If the same startup banner appears several times over, the platform is restarting the failing container in a loop and billing you for it — destroy the instance.

**`exec: creat3: not found`** — a misspelled subcommand was treated as a command to run. Check the spelling against `Arguments: help`.

**Speeds look lower than the numbers in the README** — those were measured on an Apple M4 Max, not on the card you rented. Sustained rates also settle below the first few seconds as the GPU heats up, and create3 does two keccaks per candidate so it lands near half of create2. Run `--benchmark` on the instance for its real figure.

**AMD GPUs** — the image only ships the NVIDIA ICD. ROCm needs its own OpenCL runtime inside the image plus access to `/dev/kfd` and `/dev/dri` in the container, which vast.ai does not expose on every AMD host. Start from [build/linux.md](build/linux.md) and a `rocm/dev-ubuntu-24.04` base image if you need it.
