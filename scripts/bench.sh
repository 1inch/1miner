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
# A cooldown between runs is not on its own enough. Measured on an M4 Max, a
# 30-second cooldown left create3 1.4% below its settled rate with three times
# the spread, and a benchmark started straight after a heavy GPU session read
# 284 MH/s where the settled figure was 363 — then climbed over the next three
# passes rather than falling. Alternating order cancels a drift that runs one
# way; it cannot cancel a machine still recovering from whatever ran before the
# benchmark. Hence a 60-second cooldown, and -x: whole passes run and thrown
# away before the first one that counts.
#
# Usage:
#   scripts/bench.sh [-m MODE] [-b BACKENDS] [-k KERNELS] [-p PASSES]
#                    [-w WARMUP] [-d MEASURE] [-c COOLDOWN] [-x DISCARDED]
#                    [-o FILE]
#
#   -m  mode: create2 | create3 | 1nft | profanity          (default create3)
#   -b  space-separated backends                            (default "opencl")
#   -k  space-separated kernel variants, opencl only        (default "tuned")
#   -p  passes per contender                                (default 2)
#   -w  warmup seconds, discarded within each run           (default 10)
#   -d  measured seconds                                    (default 20)
#   -c  cooldown seconds before each run                    (default 60)
#   -x  whole passes run and thrown away first              (default 1)
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
COOLDOWN=60
DISCARD=1
OUTFILE=""

# Pass-to-pass spread above this share of the mean, as a percentage, means the
# figures describe the machine's thermal state rather than the contenders.
DRIFT=3

while getopts "m:b:k:p:w:d:c:x:o:h" opt; do
    case "$opt" in
        m) MODE=$OPTARG ;;
        b) BACKENDS=$OPTARG ;;
        k) KERNELS=$OPTARG ;;
        p) PASSES=$OPTARG ;;
        w) WARMUP=$OPTARG ;;
        d) MEASURE=$OPTARG ;;
        c) COOLDOWN=$OPTARG ;;
        x) DISCARD=$OPTARG ;;
        o) OUTFILE=$OPTARG ;;
        # The comment block above, however long it grows. A fixed line range
        # went stale the first time a line was added to it.
        h) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
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

# --kernel selects one of the OpenCL keccak sources. Every other backend has
# exactly one keccak and ignores the flag, so they are given a value that parses
# and labelled for what they actually ran.
kernel_arg() {
    case "$1" in
        opencl) printf '%s' "$2" ;;
        *) printf 'tuned' ;;
    esac
}

# The last run's output, kept so that a run which reported no rate can say why.
RUNLOG=$(mktemp)
trap 'rm -f "$RUNLOG"' EXIT INT TERM

# One measurement. The miner reports a rolling-window rate while it runs and a
# post-warmup average when it stops; the latter is what a benchmark should
# quote, so `--warmup` is passed through and the `Measured:` line is read.
measure() {
    backend=$1
    kernel=$2
    total=$((WARMUP + MEASURE))
    # shellcheck disable=SC2046
    "$MINER" $(mode_args) --benchmark \
        --backend "$backend" --kernel "$(kernel_arg "$backend" "$kernel")" \
        --warmup "$WARMUP" --seconds "$total" > "$RUNLOG" 2>&1 || true
    tr '\r' '\n' < "$RUNLOG" \
        | sed -n 's/^Measured: \([0-9.]*\) MH\/s.*/\1/p' \
        | tail -1
}

# Why a run produced no rate. Everything the miner said used to go through the
# same pipe as the rate and be dropped by it, so a mode the backend refuses, a
# device that was busy and a binary too old to know the flag all came out as one
# word: FAILED. Three cooldowns and three runs to find out which, and the answer
# had been on stderr the whole time.
#
# The error line rather than the last line: clap and anyhow both put the reason
# first and boilerplate after it, so "try '--help'" is what a naive tail reports.
failure_reason() {
    tr '\r' '\n' < "$RUNLOG" | awk '
        /^[Ee]rror/ { print; found = 1; exit }
        NF { last = $0 }
        END { if (!found) print (last == "" ? "no output" : last) }
    '
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
        if [ "$backend" = opencl ]; then
            CONTENDERS="$CONTENDERS $backend:$kernel"
            continue
        fi
        # One keccak, so a second -k would rerun identical work and report it
        # as two contenders that differ by their labels alone.
        case " $CONTENDERS " in
            *" $backend:builtin "*) ;;
            *) CONTENDERS="$CONTENDERS $backend:builtin" ;;
        esac
    done
done

echo "mode=$MODE  warmup=${WARMUP}s  measured=${MEASURE}s  cooldown=${COOLDOWN}s  discarded=$DISCARD  passes=$PASSES"
echo "contenders:$CONTENDERS"
# Which binary, and how old. The candidate list below prefers
# ./target/release/1miner, which is the previous build until you rebuild, and
# benchmarking a change you have not compiled reports the change as free.
echo "miner=$MINER (built $(date -r "$MINER" '+%Y-%m-%d %H:%M'))"
if [ -d crates ] && [ -n "$(find crates kernels -type f -newer "$MINER" 2>/dev/null | head -1)" ]; then
    echo "  warning: sources are newer than this binary; rebuild, or set MINER."
fi
echo

# Whole passes thrown away, to put the machine in the state the measured passes
# will run in. This is not what -w does: -w discards the first seconds inside a
# single run, which cannot escape a thermal state that takes minutes to leave.
# Their figures are printed rather than hidden, because how far they sit from the
# measured ones is the evidence that the cooldown is long enough.
discarded=1
while [ "$discarded" -le "$DISCARD" ]; do
    echo "discarded pass $discarded"
    for c in $CONTENDERS; do
        sleep "$COOLDOWN"
        printf "  %-16s " "$c"
        speed=$(measure "${c%%:*}" "${c##*:}")
        if [ -z "$speed" ]; then
            echo "FAILED: $(failure_reason)"
        else
            echo "$speed MH/s (discarded)"
        fi
    done
    echo
    discarded=$((discarded + 1))
done

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
            echo "FAILED: $(failure_reason)"
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
    if [ -n "$mean" ]; then
        printf "  %-16s %s MH/s%s\n" "$c" "$mean" "$spread"
    fi
done

# Printing the spread is not the same as saying it is too wide to mean anything.
# "322.771 MH/s (min 284.543, max 346.768)" is not a result, and without this it
# is reported in exactly the shape of one.
drifted=""
for c in $CONTENDERS; do
    pct=$(printf '%s' "$RESULTS" | awk -v key="$c" '
        $1 == key { s += $2; n += 1; if (n == 1 || $2 < lo) lo = $2; if (n == 1 || $2 > hi) hi = $2 }
        END { if (n > 1 && s > 0) printf "%.1f", (hi - lo) * 100 * n / s }')
    if [ -n "$pct" ] && awk "BEGIN { exit !($pct > $DRIFT) }"; then
        drifted="$drifted $c ($pct%)"
    fi
done
if [ -n "$drifted" ]; then
    echo
    echo "warning: pass-to-pass spread above ${DRIFT}% for:$drifted" >&2
    echo "  Thermal drift is deciding these numbers, not the contenders. Raise -c" >&2
    echo "  (cooldown, now ${COOLDOWN}s) and -x (discarded passes, now $DISCARD)," >&2
    echo "  then re-run before quoting anything above." >&2
fi

if [ -n "$OUTFILE" ]; then
    # Record what the number depends on. An unlabelled hashrate is not
    # reproducible: driver releases move it, and so does the procedure, so the
    # flags that produced the figure go in the row beside it.
    host=$(uname -sm)
    stamp=$(date -u '+%Y-%m-%d')
    flags="-w $WARMUP -d $MEASURE -c $COOLDOWN -x $DISCARD -p $PASSES"
    [ -f "$OUTFILE" ] || printf '| date | host | mode | backend | kernel | MH/s | min-max | flags |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n' > "$OUTFILE"
    for c in $CONTENDERS; do
        mean=$(printf '%s' "$RESULTS" | awk -v key="$c" '$1 == key { s += $2; n += 1 } END { if (n) printf "%.3f", s / n }')
        range=$(printf '%s' "$RESULTS" | awk -v key="$c" '
            $1 == key { n += 1; if (n == 1 || $2 < lo) lo = $2; if (n == 1 || $2 > hi) hi = $2 }
            END { if (n > 1) printf "%.3f-%.3f", lo, hi }')
        if [ -n "$mean" ]; then
            printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
                "$stamp" "$host" "$MODE" "${c%%:*}" "${c##*:}" "$mean" "${range:--}" "$flags" >> "$OUTFILE"
        fi
    done
    echo
    echo "appended to $OUTFILE"
fi
