#!/bin/bash
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

# Run the whole script again inside the logging pipeline, before anything else
# writes or execs, and stay to wait for both halves of it.
#
# Two shorter spellings are wrong, and both fail silently. Not
# `exec 1miner "$@" | tee`: exec replaces only the pipeline's left-hand
# subshell, so the shell survived, the pipeline reported tee's status of 0, and
# control fell through to the exec at the end of the file — running the search a
# second time, unlogged, on someone else's hourly rate. And not
# `exec 1> >(tee -a "$MINER_OUTPUT") 2>&1`: that leaves tee a child of the
# container's PID 1, and the kernel SIGKILLs whatever is left in a PID namespace
# the moment its init exits, so the last lines reached neither the file nor
# stdout. A search ending on the hit it was started for lost the hit, and
# setting MINER_OUTPUT made the platform's own log lossy along with the file.
#
# Wrapping here rather than beside that exec also covers the startup warnings and
# the `sh -c '1miner self-test && 1miner ...'` form docs/vastai.md recommends,
# which leaves through the passthrough below. Nothing after this block changes
# shape: the re-entry guard sends the inner run down the ordinary exec path,
# where exec is safe because that shell is a child rather than init.
if [[ -n "${MINER_OUTPUT:-}" && -z "${MINER_LOG_WRAPPED:-}" ]]; then
    mkdir -p "$(dirname "$MINER_OUTPUT")"
    export MINER_LOG_WRAPPED=1
    # A signal that arrives while a foreground pipeline runs is deferred until
    # the pipeline finishes. With no handler installed the shell then takes the
    # default action and dies without reaching the exit below, which would
    # replace the search's own status with 130. A handler that does nothing is
    # enough; it must not be `trap '' INT`, because an ignored signal is
    # inherited and the miner would stop answering Ctrl-C.
    trap ':' INT
    "$0" "$@" 2>&1 | tee -a "$MINER_OUTPUT"
    # tee's status is the pipeline's, and it is 0 whatever the search did. This
    # is the only place the miner's own status can still be read.
    exit "${PIPESTATUS[0]}"
fi

# `bench` is in the list without being a 1miner subcommand: it is the shipped
# scripts/bench.sh, dispatched below. Left out, it would fall through to the
# passthrough and be exec'd as a command that does not exist — and it would
# reach that exec above the GPU check, which a benchmark wants ahead of its
# first cooldown rather than after its last.
is_subcommand() {
    case "$1" in
        profanity|create2|create3|1nft|self-test|bench|help) return 0 ;;
        -*) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "$#" -gt 0 ]] && ! is_subcommand "$1"; then
    exec "$@"
fi

# With no arguments, fall back to MINER_ARGS and then to the help text.
#
# The image deliberately sets no CMD: a default would arrive here as a real
# argument and MINER_ARGS would never be consulted, which is how this went
# unnoticed the first time.
if [[ "$#" -eq 0 ]]; then
    if [[ -n "${MINER_ARGS:-}" ]]; then
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
    # The harness has a help of its own, and getopts gives it only the one
    # spelling.
    bench) case "${2:-}" in -h) wants_gpu=0 ;; *) ;; esac ;;
    *) ;;
esac
for arg in "$@"; do
    case "$arg" in
        --backend=cpu) wants_gpu=0 ;;
        *) ;;
    esac
done

# Fail loudly and early rather than after the rental clock has started.
if [[ "$wants_gpu" = "1" && "${MINER_SKIP_GPU_CHECK:-0}" != "1" ]]; then
    if ! clinfo -l 2>/dev/null | grep -qi 'device'; then
        echo "warning: no OpenCL devices are visible to the container." >&2
        echo "  Pass --gpus all to docker run and check the host driver is installed." >&2
        echo "  'docker run --rm --gpus all <image> clinfo' shows what the runtime sees." >&2
        echo "  Set MINER_SKIP_GPU_CHECK=1 to silence this." >&2
    fi
fi

# Dispatched here rather than beside the passthrough above, so that `bench`
# reaches it from MINER_ARGS as well — the panels that offer an environment and
# no command line are the same ones where measuring a machine by hand is
# awkward enough to want this.
if [[ "$1" = "bench" ]]; then
    shift
    exec /usr/local/bin/1miner-bench "$@"
fi

exec /usr/local/bin/1miner "$@"
