#!/bin/sh
# Kernel-swap benchmark harness.
#
# Compares any combination of backends and kernel variants for one mode, with
# the discipline that makes GPU numbers mean something: a cooldown before every
# measurement, a warmup inside every measurement, and contenders run in
# alternating order so thermal drift cancels instead of deciding the result.
#
# Without that, a naive A/B is dominated by heat. During this project's own
# Rust-versus-C++ decision the first measurement put Rust 22% ahead purely
# because it ran first on a cold GPU.
#
# Usage:
#   scripts/bench.sh [-m MODE] [-b BACKENDS] [-k KERNELS] [-p PASSES]
#                    [-w WARMUP] [-d MEASURE] [-c COOLDOWN] [-o FILE]
#
#   -m  mode: create2 | create3 | 1nft | profanity          (default create3)
#   -b  space-separated backends                            (default "opencl")
#   -k  space-separated kernel variants                     (default "tuned")
#   -p  passes per contender                                (default 2)
#   -w  warmup seconds, discarded                           (default 10)
#   -d  measured seconds                                    (default 20)
#   -c  cooldown seconds before each run                    (default 30)
#   -o  also append a markdown table row per result to FILE
#
# Examples:
#   scripts/bench.sh -b "opencl metal"
#   scripts/bench.sh -k "tuned plain" -p 3
#   scripts/bench.sh -m create2 -b "opencl metal" -o bench-results.md
set -eu

MODE=create3
BACKENDS="opencl"
KERNELS="tuned"
PASSES=2
WARMUP=10
MEASURE=20
COOLDOWN=30
OUTFILE=""

while getopts "m:b:k:p:w:d:c:o:h" opt; do
    case "$opt" in
        m) MODE=$OPTARG ;;
        b) BACKENDS=$OPTARG ;;
        k) KERNELS=$OPTARG ;;
        p) PASSES=$OPTARG ;;
        w) WARMUP=$OPTARG ;;
        d) MEASURE=$OPTARG ;;
        c) COOLDOWN=$OPTARG ;;
        o) OUTFILE=$OPTARG ;;
        h) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "try -h" >&2; exit 2 ;;
    esac
done

MINER=${MINER:-}
if [ -z "$MINER" ]; then
    for candidate in \
        "./target/release/1miner" \
        "${CARGO_TARGET_DIR:-/nonexistent}/release/1miner" \
        "$(command -v 1miner 2>/dev/null || true)"
    do
        if [ -x "$candidate" ]; then MINER=$candidate; break; fi
    done
fi
if [ -z "$MINER" ] || [ ! -x "$MINER" ]; then
    echo "cannot find the 1miner binary; build it or set MINER=/path/to/1miner" >&2
    exit 1
fi

# A zero deployer is fine: --benchmark scores nothing, so the address is never
# used. Profanity still needs a well-formed public key, so the generator point
# stands in.
ZERO_ADDR=0x0000000000000000000000000000000000000000
GENERATOR_PUBKEY=79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

mode_args() {
    case "$MODE" in
        create2)   echo "create2 --deployer $ZERO_ADDR --init-code 0x00" ;;
        create3)   echo "create3 --deployer $ZERO_ADDR" ;;
        1nft)      echo "1nft --deployer $ZERO_ADDR --mint-for $ZERO_ADDR" ;;
        profanity) echo "profanity --public-key $GENERATOR_PUBKEY" ;;
        *) echo "unknown mode: $MODE" >&2; exit 2 ;;
    esac
}

# One measurement. The miner prints a rolling speed line; the last one is the
# settled figure. Warmup is spent inside the same process so kernel compilation
# and ramp-up are excluded from the number we keep.
measure() {
    backend=$1
    kernel=$2
    total=$((WARMUP + MEASURE))
    # shellcheck disable=SC2046
    "$MINER" $(mode_args) --benchmark \
        --backend "$backend" --kernel "$kernel" --seconds "$total" 2>&1 \
        | tr '\r' '\n' \
        | sed -n 's/.*Speed: \([0-9.]*\) MH\/s.*/\1/p' \
        | tail -1
}

DEVICE=$("$MINER" self-test --backend "$(echo "$BACKENDS" | cut -d' ' -f1)" >/dev/null 2>&1 \
    && echo ok || echo unverified)
if [ "$DEVICE" = "unverified" ]; then
    echo "warning: self-test did not pass on the first backend." >&2
    echo "  A fast but wrong kernel is the failure this project guards against;" >&2
    echo "  fix correctness before trusting any number below." >&2
    echo >&2
fi

CONTENDERS=""
for backend in $BACKENDS; do
    for kernel in $KERNELS; do
        CONTENDERS="$CONTENDERS $backend:$kernel"
    done
done

echo "mode=$MODE  warmup=${WARMUP}s  measured=${MEASURE}s  cooldown=${COOLDOWN}s  passes=$PASSES"
echo "contenders:$CONTENDERS"
echo

RESULTS=""
pass=1
while [ "$pass" -le "$PASSES" ]; do
    # Reverse the order on even passes so being second is not a penalty that
    # always lands on the same contender.
    ordered=$CONTENDERS
    if [ $((pass % 2)) -eq 0 ]; then
        ordered=""
        for c in $CONTENDERS; do ordered="$c $ordered"; done
    fi

    echo "pass $pass"
    for c in $ordered; do
        backend=${c%%:*}
        kernel=${c##*:}
        sleep "$COOLDOWN"
        printf "  %-16s " "$c"
        speed=$(measure "$backend" "$kernel")
        if [ -z "$speed" ]; then
            echo "FAILED (no speed reported)"
            continue
        fi
        echo "$speed MH/s"
        RESULTS="$RESULTS$c $speed
"
    done
    pass=$((pass + 1))
done

echo
echo "mean per contender:"
for c in $CONTENDERS; do
    mean=$(printf '%s' "$RESULTS" | awk -v key="$c" '$1 == key { s += $2; n += 1 } END { if (n) printf "%.3f", s / n }')
    spread=$(printf '%s' "$RESULTS" | awk -v key="$c" '
        $1 == key { if (n == 0 || $2 < lo) lo = $2; if ($2 > hi) hi = $2; n += 1 }
        END { if (n > 1) printf " (min %.3f, max %.3f)", lo, hi }')
    [ -n "$mean" ] && printf "  %-16s %s MH/s%s\n" "$c" "$mean" "$spread"
done

if [ -n "$OUTFILE" ]; then
    # Record what the number depends on. An unlabelled hashrate is not
    # reproducible: driver releases move it.
    host=$(uname -sm)
    stamp=$(date -u '+%Y-%m-%d')
    [ -f "$OUTFILE" ] || printf '| date | host | mode | backend | kernel | MH/s |\n| --- | --- | --- | --- | --- | --- |\n' > "$OUTFILE"
    for c in $CONTENDERS; do
        mean=$(printf '%s' "$RESULTS" | awk -v key="$c" '$1 == key { s += $2; n += 1 } END { if (n) printf "%.3f", s / n }')
        [ -n "$mean" ] && printf '| %s | %s | %s | %s | %s | %s |\n' \
            "$stamp" "$host" "$MODE" "${c%%:*}" "${c##*:}" "$mean" >> "$OUTFILE"
    done
    echo
    echo "appended to $OUTFILE"
fi
