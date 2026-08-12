# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM rust:1-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        opencl-headers \
        ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Manifests first so dependency compilation is cached independently of source
# edits. The dummy sources are replaced in the next layer.
COPY Cargo.toml Cargo.lock ./
COPY crates/miner-core/Cargo.toml crates/miner-core/
COPY crates/miner-backend/Cargo.toml crates/miner-backend/
COPY crates/miner-cli/Cargo.toml crates/miner-cli/
RUN mkdir -p crates/miner-core/src crates/miner-backend/src crates/miner-cli/src \
    && echo "" > crates/miner-core/src/lib.rs \
    && echo "" > crates/miner-backend/src/lib.rs \
    && echo "fn main() {}" > crates/miner-cli/src/main.rs \
    && cargo build --release --locked 2>/dev/null || true

COPY crates crates
COPY kernels kernels
# Cargo skips rebuilding when only mtimes look stale, so force the real sources.
RUN touch crates/*/src/lib.rs crates/miner-cli/src/main.rs \
    && cargo build --release --locked --bin 1miner \
    && strip target/release/1miner

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="1miner" \
      org.opencontainers.image.description="GPU miner for vanity Ethereum addresses: profanity, CREATE2, CREATE3 and 1inch Address NFT" \
      org.opencontainers.image.source="https://github.com/1inch/1miner" \
      org.opencontainers.image.licenses="MIT"

# ocl-icd-libopencl1 is the ICD loader the binary links against; clinfo earns
# its place by making "no devices found" diagnosable on a rented machine.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        clinfo \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The NVIDIA container runtime mounts libnvidia-opencl.so.1 into the container
# but does not register it with the ICD loader, so the vendor file has to ship
# in the image: https://github.com/NVIDIA/nvidia-container-toolkit/issues/682
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# compute exposes the OpenCL driver, utility exposes nvidia-smi.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Kernels are embedded in the binary, so unlike profanity2 there are no .cl
# files to keep beside it and no working-directory requirement.
COPY --from=build /src/target/release/1miner /usr/local/bin/1miner
COPY LICENSE /usr/share/doc/1miner/LICENSE
# The entrypoint needs bash for the MINER_OUTPUT redirect, which bookworm-slim
# has and a slimmer base such as alpine would not.
COPY docker/entrypoint.sh /usr/local/bin/1miner-entrypoint
# `1miner bench` on a rented box, where the image is everything the machine
# has. Deciding whether an offer is worth its hourly rate otherwise means
# typing the whole procedure into a template field.
COPY scripts/bench.sh /usr/local/bin/1miner-bench
RUN chmod +x /usr/local/bin/1miner-entrypoint /usr/local/bin/1miner-bench /usr/local/bin/1miner \
    && mkdir -p /workspace

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/1miner-entrypoint"]
