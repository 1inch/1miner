#!/bin/sh
# Container entrypoint for 1miner.
#
# Arguments pass straight through to the binary. Two conveniences exist for
# rented GPU boxes where setting a command line is awkward: arguments can come
# from MINER_ARGS, and output can be tee'd somewhere that outlives the run.
#
# Anything that is not a 1miner subcommand or flag is executed as a plain
# command, so `docker run ... clinfo`, `nvidia-smi` and `sh` still work for
# debugging. That check is an explicit allowlist of subcommands rather than a
# "does it start with a dash" test, because 1miner's own subcommands do not.
set -eu

is_subcommand() {
    case "$1" in
        profanity|create2|create3|1nft|self-test|help) return 0 ;;
        -*) return 0 ;;
        *) return 1 ;;
    esac
}

if [ "$#" -gt 0 ] && ! is_subcommand "$1"; then
    exec "$@"
fi

# With no arguments, fall back to MINER_ARGS and then to the help text.
#
# The image deliberately sets no CMD: a default would arrive here as a real
# argument and MINER_ARGS would never be consulted, which is how this went
# unnoticed the first time.
if [ "$#" -eq 0 ]; then
    if [ -n "${MINER_ARGS:-}" ]; then
        # Deliberately unquoted: MINER_ARGS holds a whole argument list.
        # shellcheck disable=SC2086
        set -- $MINER_ARGS
    else
        set -- --help
    fi
fi

# Only worth checking when we are about to mine. Printing a GPU warning in
# front of --help is noise.
wants_gpu=1
case "${1:-}" in
    -h|--help|-V|--version|help|"") wants_gpu=0 ;;
esac
for arg in "$@"; do
    case "$arg" in
        --backend=cpu) wants_gpu=0 ;;
    esac
done

# Fail loudly and early rather than after the rental clock has started.
if [ "$wants_gpu" = "1" ] && [ "${MINER_SKIP_GPU_CHECK:-0}" != "1" ]; then
    if ! clinfo -l 2>/dev/null | grep -qi 'device'; then
        echo "warning: no OpenCL devices are visible to the container." >&2
        echo "  Pass --gpus all to docker run and check the host driver is installed." >&2
        echo "  'docker run --rm --gpus all <image> clinfo' shows what the runtime sees." >&2
        echo "  Set MINER_SKIP_GPU_CHECK=1 to silence this." >&2
    fi
fi

if [ -n "${MINER_OUTPUT:-}" ]; then
    mkdir -p "$(dirname "$MINER_OUTPUT")"
    exec /usr/local/bin/1miner "$@" 2>&1 | tee -a "$MINER_OUTPUT"
fi

exec /usr/local/bin/1miner "$@"
