#!/bin/sh
# Whole-machine benchmark: every mode, one figure each.
#
# Answers the question renting hardware asks — what is this machine worth —
# where scripts/bench-variants.sh answers a different one, which backend or
# kernel is faster on a machine you already have. One backend against every
# mode, so the output is a handful of figures you can set beside another
# machine's.
#
# The figures are not comparable *between* the modes below. create3 and 1nft
# hash twice per candidate where create2 hashes once, and profanity does
# secp256k1 point arithmetic as well. Compare each mode against the same mode
# on the other machine.
#
# This ships inside the published container image and runs there as
# `1miner bench`, which is what makes it useful on a rented box: the image is
# the only thing on the machine. See docs/vastai.md.
#
# Usage:
#   scripts/bench.sh [--fast|--balanced|--accurate] [overrides]
#
# Profiles:
#   --fast      the default, and cheap enough to run on every offer you rent.
#               No discarded pass, so a cold machine reads high; see below.
#   --balanced  a discarded pass first, so what is reported is a settled rate.
#   --accurate  two measured passes as well, with the spread printed.
#   --custom    no profile of its own; the overrides below imply it anyway.
#
# Overrides, in any order, with or without a profile:
#   -c SECONDS   idle before each run                       (cooldown)
#   -x COUNT     whole passes run and thrown away first
#   -w SECONDS   excluded from the front of each run        (warmup)
#   -d SECONDS   measured seconds per run
#   -p COUNT     measured passes per mode
#   -M "MODES"   modes to run, space separated
#   -b BACKEND   opencl | metal | cpu
#   -k KERNEL    tuned | plain, opencl only
#   -T           run without the self-test gate
#   -o FILE      append a markdown row per mode
#   -h           this text
#
# Examples:
#   scripts/bench.sh                            # every mode, a few minutes
#   scripts/bench.sh --accurate -o rented.md
#   scripts/bench.sh --fast -M "create3 profanity"
#
# On --fast and cold machines. A freshly started instance has an idle GPU, so
# the first mode in the list is measured on one and reads high; without a
# discarded pass nothing absorbs that. It is a bias every machine running this
# script gets in the same place, so --fast figures still rank offers against
# each other. They are not settled rates, and mixing them with --balanced ones
# in the same table compares the procedures rather than the machines.
set -eu

MODES="create2 create3 1nft profanity"
BACKEND=opencl
KERNEL=tuned
SELFTEST=1
OUTFILE=""

# profanity allocates and initialises millions of points before it hashes
# anything, which a warmup sized for the salt modes does not cover: the run
# then reports a rate diluted by its own start-up. Cross-mode comparison is
# meaningless here anyway, so the mode gets a longer warmup rather than the
# whole benchmark paying for one.
PROFANITY_WARMUP=15

profile() {
    PROFILE=$1
    case "$1" in
        fast)     COOLDOWN=30; DISCARD=0; WARMUP=5;  MEASURE=15; PASSES=1 ;;
        balanced) COOLDOWN=60; DISCARD=1; WARMUP=10; MEASURE=30; PASSES=1 ;;
        accurate) COOLDOWN=60; DISCARD=1; WARMUP=10; MEASURE=30; PASSES=2 ;;
    esac
}
profile fast

usage() {
    # The comment block above, however long it grows. A fixed line range went
    # stale the first time a line was added to it.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

# Two passes over the arguments, so a profile and an override commute. Read in
# one pass, `--balanced -c 30` and `-c 30 --balanced` would mean different
# things, and the second of them silently.
for arg in "$@"; do
    case "$arg" in
        --fast|--balanced|--accurate) profile "${arg#--}" ;;
        --custom) PROFILE=custom ;;
        -h|--help) usage; exit 0 ;;
    esac
done

overridden=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fast|--balanced|--accurate|--custom) ;;
        -c) COOLDOWN=$2; overridden=1; shift ;;
        -x) DISCARD=$2;  overridden=1; shift ;;
        -w) WARMUP=$2;   overridden=1; shift ;;
        -d) MEASURE=$2;  overridden=1; shift ;;
        -p) PASSES=$2;   overridden=1; shift ;;
        -M) MODES=$2;   shift ;;
        -b) BACKEND=$2; shift ;;
        -k) KERNEL=$2;  shift ;;
        -o) OUTFILE=$2; shift ;;
        -T) SELFTEST=0 ;;
        *) echo "unknown option: $1" >&2; echo "try -h" >&2; exit 2 ;;
    esac
    shift
done
# Said out loud, because the profile name is what a reader of the log will take
# the procedure from, and an overridden one no longer describes it.
[ -z "$overridden" ] || PROFILE="custom (from $PROFILE)"

MINER=${MINER:-}
if [ -z "$MINER" ]; then
    # CARGO_TARGET_DIR before ./target/release, because when it is set that is
    # where cargo just wrote and ./target holds whatever was last built before
    # it was set. The other order benchmarked a two-day-old binary here and
    # reported the change under test as costing nothing.
    for candidate in \
        "${CARGO_TARGET_DIR:-/nonexistent}/release/1miner" \
        "./target/release/1miner" \
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
    case "$1" in
        create2)   echo "create2 --deployer $ZERO_ADDR --init-code 0x00" ;;
        create3)   echo "create3 --deployer $ZERO_ADDR" ;;
        1nft)      echo "1nft --deployer $ZERO_ADDR --mint-for $ZERO_ADDR" ;;
        profanity) echo "profanity --public-key $GENERATOR_PUBKEY" ;;
        *) echo "unknown mode: $1" >&2; exit 2 ;;
    esac
}

mode_warmup() {
    if [ "$1" = profanity ] && [ "$WARMUP" -lt "$PROFANITY_WARMUP" ]; then
        echo "$PROFANITY_WARMUP"
    else
        echo "$WARMUP"
    fi
}

for mode in $MODES; do mode_args "$mode" > /dev/null; done

RUNLOG=$(mktemp)
trap 'rm -f "$RUNLOG"' EXIT INT TERM

# One measurement. The miner reports a rolling-window rate while it runs and a
# post-warmup average when it stops; the latter is what a benchmark should
# quote, so `--warmup` is passed through and the `Measured:` line is read.
measure() {
    warmup=$(mode_warmup "$1")
    # shellcheck disable=SC2046
    "$MINER" $(mode_args "$1") --benchmark \
        --backend "$BACKEND" --kernel "$KERNEL" \
        --warmup "$warmup" --seconds "$((warmup + MEASURE))" > "$RUNLOG" 2>&1 || true
    tr '\r' '\n' < "$RUNLOG" \
        | sed -n 's/^Measured: \([0-9.]*\) MH\/s.*/\1/p' \
        | tail -1
}

# Why a run produced no rate. Everything the miner said used to go through the
# same pipe as the rate and be dropped by it, so a mode the backend refuses, a
# device that was busy and a binary too old to know the flag all came out as
# one word: FAILED.
#
# The error line rather than the last line: clap and anyhow both put the reason
# first and boilerplate after it, so "try '--help'" is what a naive tail
# reports.
failure_reason() {
    tr '\r' '\n' < "$RUNLOG" | awk '
        /^[Ee]rror/ { print; found = 1; exit }
        NF { last = $0 }
        END { if (!found) print (last == "" ? "no output" : last) }
    '
}

runs=0
for mode in $MODES; do runs=$((runs + 1)); done
runs=$((runs * (DISCARD + PASSES)))
estimate=$(( (runs * (COOLDOWN + WARMUP + MEASURE) + 59) / 60 ))

echo "1miner bench"
echo "  version    $("$MINER" --version 2>/dev/null || echo unknown)"
# Which file, and how old. Benchmarking a change you have not compiled reports
# the change as free, and the path is here beside the timestamp because more
# than one target directory can exist on a machine that sets CARGO_TARGET_DIR.
echo "  binary     $MINER (built $(date -r "$MINER" '+%Y-%m-%d %H:%M' 2>/dev/null || echo unknown))"
if [ -d crates ] && [ -n "$(find crates kernels -type f -newer "$MINER" 2>/dev/null | head -1)" ]; then
    echo "             warning: sources are newer than this binary; rebuild, or set MINER."
fi
echo "  profile    $PROFILE"
echo "  procedure  cooldown ${COOLDOWN}s, warmup ${WARMUP}s, measured ${MEASURE}s, $PASSES pass(es), $DISCARD discarded"
echo "  backend    $BACKEND (kernel $KERNEL)"
echo "  modes      $MODES"
echo "  date       $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "  estimate   about $estimate minute(s), plus start-up"
echo

# What the machine is, as far as it can be asked. The GPU dominates a figure
# here and gets its own block below, but the host is worth recording too: it
# feeds every device its rounds and re-derives every hit on the CPU, it is the
# whole story under `-b cpu`, and on a rented box it is routinely a slice of a
# chip rather than the chip named on the offer.
echo "Machine"
cpu=unknown
if command -v sw_vers > /dev/null 2>&1; then
    cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)
    cores=$(sysctl -n hw.ncpu 2>/dev/null || echo '?')
    membytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    os="$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
else
    # `model name` is the x86 spelling and the one a rented box will have.
    # `Model` is what some ARM kernels write instead, lscpu knows a few more,
    # and the architecture is a poor last answer but a true one — `uname -p`
    # is not, since Linux answers "unknown" to it.
    for probe in \
        "$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1)" \
        "$(sed -n 's/^Model[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1)" \
        "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)" \
        "$(uname -m)"
    do
        # lscpu answers "-" for a chip it cannot name, which is an absence
        # wearing the clothes of a value and would otherwise be printed as one.
        case "$probe" in
            '' | - | unknown) continue ;;
            *) cpu=$probe; break ;;
        esac
    done
    cores=$(nproc 2>/dev/null || echo '?')
    memkb=$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\) kB$/\1/p' /proc/meminfo 2>/dev/null)
    membytes=$(( ${memkb:-0} * 1024 ))
    os=$(sed -n 's/^PRETTY_NAME="\(.*\)"$/\1/p' /etc/os-release 2>/dev/null)
    [ -n "$os" ] || os=$(uname -s)
    os="$os, kernel $(uname -r)"
fi
echo "  cpu        $cpu ($cores cores)"
echo "  memory     $(awk -v b="$membytes" 'BEGIN { if (b > 0) printf "%.0f GB", b / 1073741824; else print "unknown" }')"
echo "  os         $os"
echo "  arch       $(uname -sm)"

# A laptop on battery is a different machine. macOS clocks the GPU down hard
# with the cable out, and nothing else in this output says so: the first run of
# this script read 189 MH/s on profanity against 271 for the identical command
# a few minutes later, which was taken for a bug in the harness until someone
# noticed the cable. A rented box is always on mains, so this costs nothing
# there and is worth the two lines everywhere else.
on_battery() {
    if command -v pmset > /dev/null 2>&1; then
        pmset -g batt 2>/dev/null | grep -q "Battery Power"
        return $?
    fi
    for supply in /sys/class/power_supply/A*/online; do
        if [ -r "$supply" ] && [ "$(cat "$supply")" = "0" ]; then
            return 0
        fi
    done
    return 1
}
if on_battery; then
    echo "warning: this machine is running on battery, so its GPU is clocked down."
    echo "  Plug it in before measuring anything."
    echo
fi

# The driver version belongs in the record beside the rate, because a hashrate
# moves with driver releases and a rented machine is the one place you did not
# choose which driver you got. nvidia-smi is in the image; on other hosts this
# is simply absent.
if command -v nvidia-smi > /dev/null 2>&1; then
    smi_gpus=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$smi_gpus" ]; then
        printf '%s\n' "$smi_gpus" | awk '{ printf "  gpu%-8s%s\n", NR - 1, $0 }'
    else
        echo "  gpu        nvidia-smi reports no device"
    fi
fi
echo

# The self-test gate. On hardware you rent this is the whole reason to run
# anything before mining: a bad driver or a miscompiled kernel derives
# plausible-looking wrong addresses and nothing else says so. Benchmarking such
# a box for the next several minutes, and paying for it, is not worth doing.
if [ "$SELFTEST" = "1" ]; then
    printf 'Checking the device against the CPU reference... '
    if "$MINER" self-test --backend "$BACKEND" > "$RUNLOG" 2>&1; then
        echo "ok"
    else
        echo "FAILED"
        echo
        sed 's/^./  &/' < "$RUNLOG"
        echo
        echo "Nothing measured on this device would mean anything: it either derives" >&2
        echo "addresses differently from the CPU reference, which produces plausible" >&2
        echo "wrong ones, or has no usable device at all, which produces no rate. The" >&2
        echo "checks above say which. On a rented machine, destroy the instance and" >&2
        echo "rent another. Pass -T to benchmark it anyway." >&2
        exit 1
    fi
    echo
fi

DEVICES=""
NDEVICES=0

# The measured passes run the modes in a fixed order, where bench-variants.sh
# alternates its contenders. The difference is what is being compared: there,
# two contenders race each other and being second is a penalty that must not
# always land on the same one. Here nothing races, and the comparison is
# against another machine running this same script — so create3 always third
# puts the same thermal position on both sides of that comparison, and
# shuffling would only add noise to it.
RESULTS=""
pass=1
total=$((DISCARD + PASSES))
while [ "$pass" -le "$total" ]; do
    if [ "$pass" -le "$DISCARD" ]; then
        # Printed rather than hidden: how far a discarded figure sits from the
        # measured ones is the evidence that the cooldown was long enough.
        echo "discarded pass $pass"
    elif [ "$PASSES" -gt 1 ]; then
        echo "pass $((pass - DISCARD))"
    else
        echo "measuring"
    fi

    for mode in $MODES; do
        sleep "$COOLDOWN"
        printf '  %-10s ' "$mode"
        speed=$(measure "$mode")
        if [ -z "$speed" ]; then
            echo "FAILED: $(failure_reason)"
            continue
        fi
        if [ "$NDEVICES" = "0" ]; then
            # The miner's own device line, whole: it already carries the name,
            # the memory, the compute units and the driver version, and this is
            # the list that respects --skip where nvidia-smi's does not.
            DEVICES=$(sed -n 's/^  \(GPU[0-9]*: .*\)$/\1/p' "$RUNLOG")
            NDEVICES=$(printf '%s' "$DEVICES" | grep -c . || true)
            NDEVICES=${NDEVICES:-0}
        fi
        if [ "$pass" -le "$DISCARD" ]; then
            echo "$speed MH/s (discarded)"
        else
            echo "$speed MH/s"
            RESULTS="$RESULTS$mode $speed
"
        fi
    done
    echo
    pass=$((pass + 1))
done

if [ "$NDEVICES" -gt 0 ]; then
    echo "Devices the miner used:"
    printf '%s\n' "$DEVICES" | sed 's/^/  /'
    echo
fi

# The summary is what gets copied out of an instance log, so it repeats the
# figures rather than making a reader scroll back through the passes.
echo "Summary"
if [ "$NDEVICES" -gt 1 ]; then
    printf '  %-10s %12s %12s\n' mode "MH/s" "per GPU"
else
    printf '  %-10s %12s\n' mode "MH/s"
fi
for mode in $MODES; do
    mean=$(printf '%s' "$RESULTS" | awk -v key="$mode" '$1 == key { s += $2; n += 1 } END { if (n) printf "%.3f", s / n }')
    if [ -z "$mean" ]; then
        printf '  %-10s %12s\n' "$mode" "-"
        continue
    fi
    spread=$(printf '%s' "$RESULTS" | awk -v key="$mode" '
        $1 == key { n += 1; if (n == 1 || $2 < lo) lo = $2; if (n == 1 || $2 > hi) hi = $2 }
        END { if (n > 1) printf "  (min %.3f, max %.3f)", lo, hi }')
    if [ "$NDEVICES" -gt 1 ]; then
        per=$(awk -v m="$mean" -v n="$NDEVICES" 'BEGIN { printf "%.3f", m / n }')
        printf '  %-10s %12s %12s%s\n' "$mode" "$mean" "$per" "$spread"
    else
        printf '  %-10s %12s%s\n' "$mode" "$mean" "$spread"
    fi
done

if [ "$DISCARD" -eq 0 ]; then
    echo
    echo "  No pass was discarded, so these read high on a machine that was idle when"
    echo "  the benchmark started. Good enough to rank one offer against another run"
    echo "  the same way; use --balanced for a rate worth quoting on its own."
fi

if [ -n "$OUTFILE" ]; then
    # Record what the number depends on. An unlabelled hashrate is not
    # reproducible: driver releases move it, and so does the procedure, so the
    # flags that produced the figure go in the row beside it.
    first=$(printf '%s' "$DEVICES" | head -1)
    gpus=$(printf '%s' "$first" | sed 's/^GPU[0-9]*: //; s/,.*//')
    if [ "$NDEVICES" -gt 1 ]; then
        gpus="$NDEVICES x $gpus"
    fi
    # A rate without the driver behind it cannot be compared against the same
    # machine six months later, which is most of what a recorded table is for.
    driver=$(printf '%s' "$first" | sed -n 's/.*, driver \(.*\)$/\1/p')
    stamp=$(date -u '+%Y-%m-%d')
    flags="-c $COOLDOWN -x $DISCARD -w $WARMUP -d $MEASURE -p $PASSES"
    [ -f "$OUTFILE" ] || printf '| date | gpu | driver | os | mode | backend | kernel | MH/s | profile | flags |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n' > "$OUTFILE"
    for mode in $MODES; do
        mean=$(printf '%s' "$RESULTS" | awk -v key="$mode" '$1 == key { s += $2; n += 1 } END { if (n) printf "%.3f", s / n }')
        if [ -n "$mean" ]; then
            printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
                "$stamp" "${gpus:-unknown}" "${driver:--}" "$os" "$mode" \
                "$BACKEND" "$KERNEL" "$mean" "$PROFILE" "$flags" >> "$OUTFILE"
        fi
    done
    echo
    echo "appended to $OUTFILE"
fi
