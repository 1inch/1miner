#!/bin/sh
# Turn a profanity offset into the private key of the address it found.
#
#   final = (seed private key + offset) mod n
#
# This is the one step in the whole flow that nothing checks for you, and its
# mistakes are all silent: a sum printed 63 characters long because the leading
# zero was dropped, an offset from a run against a different seed key, or the
# openssl text header pasted in where the private key should be. Each produces a
# perfectly well-formed key, for an address nobody can spend from.
#
# So the sum is computed twice, by bc and by python3, and the address of the
# result is derived from the key itself and printed. Pass --address and the run
# fails unless it lands exactly on the address the miner reported.
#
# Usage:
#   scripts/profanity-final-key.sh --offset OFFSET [SEED_PRIVATE_KEY]
#                                  [--address ADDRESS]
#
#   --offset OFFSET    the offset 1miner printed, with or without 0x
#   --address ADDRESS  the address 1miner printed beside it, checked not assumed
#   SEED_PRIVATE_KEY   defaults to $PROFANITY_PK, which profanity-keygen.sh sets
#
# Examples:
#   scripts/profanity-final-key.sh --offset 0x00000000002d76a49f72887c449dbcaa169768e4f1f957e16de3deba9a3bfbef
#   scripts/profanity-final-key.sh --offset 0x2d76a4... --address 0x0000000028f9357e9E3e0fB18cb43f0122585B90
#
# The key is the only thing on stdout, so `key=$(scripts/profanity-final-key.sh
# --offset 0x...)` works; everything else goes to stderr. Prefer $PROFANITY_PK
# to an argument. An argument leaves the key in the shell history, and in the
# process list for as long as the run takes, where any other local user can read
# it -- /proc/PID/cmdline on Linux, ps on macOS. The sums below go over a pipe
# precisely to stay out of that list, so the entry point should not undo it.
set -eu

N=fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
ZERO=0000000000000000000000000000000000000000000000000000000000000000
# keccak256 of nothing at all, the known answer that says a digest offered under
# that name really is Keccak-256 and not SHA3-256, which differs only in padding.
KECCAK_OF_EMPTY=c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470

die() {
    printf 'profanity-final-key: %s\n' "$1" >&2
    exit 1
}

usage() {
    # The comment block above, however long it grows, as bench.sh does it: the
    # fixed line range this replaces went stale the moment a line was added.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

OFFSET=""
SEED=""
ADDRESS=""
while [ $# -gt 0 ]; do
    case $1 in
        --offset)
            [ $# -ge 2 ] || die "--offset needs a value"
            OFFSET=$2
            shift 2
            ;;
        --offset=*)
            OFFSET=${1#*=}
            shift
            ;;
        --address)
            [ $# -ge 2 ] || die "--address needs a value"
            ADDRESS=$2
            shift 2
            ;;
        --address=*)
            ADDRESS=${1#*=}
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            printf 'unknown option %s; try -h\n' "$1" >&2
            exit 2
            ;;
        *)
            [ -z "$SEED" ] || {
                printf 'unexpected argument %s; try -h\n' "$1" >&2
                exit 2
            }
            SEED=$1
            shift
            ;;
    esac
done

upper() {
    printf '%s' "$1" | tr 'a-f' 'A-F'
}

pad64() {
    printf '%064s' "$1" | tr ' ' '0'
}

# One hex input, normalised: no 0x, lower case, nothing but hex digits, padded
# to 32 bytes. Length is checked against 64 rather than truncated, because the
# input most likely to be too long is openssl's text header with a private key
# somewhere inside it.
hex64() {
    _label=$1
    _value=${2#0x}
    _value=${_value#0X}
    _value=$(printf '%s' "$_value" | tr 'A-F' 'a-f')
    case $_value in
        '') die "$_label is empty" ;;
        *[!0-9a-f]*) die "$_label is not hexadecimal: $2" ;;
    esac
    if [ ${#_value} -gt 64 ]; then
        die "$_label is ${#_value} hex digits long; a 256-bit number has 64"
    fi
    pad64 "$_value"
}

if command -v bc >/dev/null 2>&1; then HAVE_BC=1; else HAVE_BC=0; fi
if command -v python3 >/dev/null 2>&1; then HAVE_PY=1; else HAVE_PY=0; fi
if [ "$HAVE_BC" = 0 ] && [ "$HAVE_PY" = 0 ]; then
    die "neither bc nor python3 is installed, and one of them has to add the two numbers"
fi

# (a + b) mod n. Both engines run when both are present, because this sum is the
# value that spends the funds: bc needs upper-case input and prints no leading
# zeros, python3 needs neither, and a slip in either direction still looks like
# a private key. Numbers go in over a pipe, never as arguments, so a private key
# does not appear in the process list.
add_mod_n() {
    _by_bc=""
    _by_py=""
    if [ "$HAVE_BC" = 1 ]; then
        _by_bc=$(printf 'ibase=16;obase=10;(%s + %s) %% %s\n' \
            "$(upper "$1")" "$(upper "$2")" "$(upper "$N")" |
            bc | tr -d '\\\n' | tr 'A-F' 'a-f')
        _by_bc=$(pad64 "$_by_bc")
    fi
    if [ "$HAVE_PY" = 1 ]; then
        _by_py=$(printf '%s %s %s\n' "$1" "$2" "$N" | python3 -c '
import sys

a, b, n = (int(x, 16) for x in sys.stdin.read().split())
print("%064x" % ((a + b) % n))')
    fi
    if [ -n "$_by_bc" ] && [ -n "$_by_py" ] && [ "$_by_bc" != "$_by_py" ]; then
        die "bc and python3 disagree about the sum, so one of them is wrong; use neither"
    fi
    _sum=${_by_bc:-$_by_py}
    if [ ${#_sum} -ne 64 ]; then
        die "the sum came out ${#_sum} hex digits long instead of 64"
    fi
    printf '%s' "$_sum"
}

# Keccak-256 of stdin, from whichever tool has one that passes the known-answer
# test above. An empty result means the address cannot be derived here.
keccak256() {
    case $KECCAK in
        openssl) openssl dgst -keccak-256 2>/dev/null | sed 's/^.*= *//' ;;
        cast) cast keccak "0x$(od -An -v -tx1 | tr -d ' \n')" | sed 's/^0x//' ;;
        *) : ;;
    esac
}

KECCAK=""
if command -v openssl >/dev/null 2>&1 &&
    [ "$(printf '' | openssl dgst -keccak-256 2>/dev/null | sed 's/^.*= *//')" = "$KECCAK_OF_EMPTY" ]; then
    KECCAK=openssl
elif command -v cast >/dev/null 2>&1 &&
    [ "$(cast keccak 0x 2>/dev/null)" = "0x$KECCAK_OF_EMPTY" ]; then
    KECCAK=cast
fi

# The address a private key controls, worked out from the key rather than read
# back from the miner: openssl derives the public point, Keccak-256 of its 64
# bytes gives the address in the low 20. Prints nothing if the tools for it are
# missing, which is a reason to say so rather than to guess.
address_of() {
    if [ -z "$KECCAK" ] ||
        ! command -v openssl >/dev/null 2>&1 ||
        ! command -v xxd >/dev/null 2>&1; then
        return 0
    fi
    # A SEC1 EC private key wrapping the 32 bytes, which openssl will complete
    # with the public half. The key reaches openssl on stdin, not in argv.
    _pub=$(printf '302e0201010420%sa00706052b8104000a' "$1" | xxd -r -p |
        openssl ec -inform DER -text -noout 2>/dev/null |
        sed -n '/^pub:/,/^ASN1 OID:/p' | sed '1d;$d' | tr -cd '0-9a-f')
    case $_pub in
        04*) ;;
        *) return 0 ;;
    esac
    [ ${#_pub} -eq 130 ] || return 0
    _hash=$(printf '%s' "${_pub#04}" | xxd -r -p | keccak256)
    [ ${#_hash} -eq 64 ] || return 0
    printf '0x%s' "$(printf '%s' "$_hash" | cut -c 25-64)"
}

[ -n "$OFFSET" ] || {
    printf 'profanity-final-key: --offset is required; try -h\n' >&2
    exit 2
}
if [ -z "$SEED" ]; then
    SEED=${PROFANITY_PK-}
    [ -n "$SEED" ] || die "no seed private key: pass it as an argument, or set \$PROFANITY_PK as scripts/profanity-keygen.sh does"
fi

SEED=$(hex64 "the seed private key" "$SEED")
OFFSET=$(hex64 "--offset" "$OFFSET")

# Reducing the seed key modulo n has to leave it alone; if it does not, the
# number is at or past the group order and is not a private key at all.
SEED_REDUCED=$(add_mod_n "$SEED" "$ZERO")
if [ "$SEED" = "$ZERO" ] || [ "$SEED_REDUCED" != "$SEED" ]; then
    die "the seed private key is not a valid secp256k1 key: it must be below fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141 and above zero"
fi

FINAL=$(add_mod_n "$SEED" "$OFFSET")
[ "$FINAL" != "$ZERO" ] || die "the sum is zero, which is not a private key; check the offset"

DERIVED=$(address_of "$FINAL")
if [ -n "$ADDRESS" ]; then
    WANTED=$(printf '%s' "${ADDRESS#0x}" | tr 'A-F' 'a-f')
    case $WANTED in
        '' | *[!0-9a-f]*) die "--address is not an address: $ADDRESS" ;;
    esac
    [ ${#WANTED} -eq 40 ] || die "--address has ${#WANTED} hex digits; an address has 40"
    [ -n "$DERIVED" ] || die "cannot check --address here: no Keccak-256 available (openssl 3.2 or later, or foundry's cast) and no xxd"
    if [ "0x$WANTED" != "$DERIVED" ]; then
        printf 'profanity-final-key: the key adds up but lands on %s, not on %s.\n' \
            "$DERIVED" "$ADDRESS" >&2
        die "the seed key and the offset are not from the same run, or one of them was mistyped"
    fi
fi

printf '%s\n' "$FINAL"

{
    printf '\nPrivate key for the mined address:\n\n'
    printf '  private key  %s\n' "$FINAL"
    if [ -n "$ADDRESS" ]; then
        printf '  address      %s  (matches --address)\n' "$ADDRESS"
    elif [ -n "$DERIVED" ]; then
        printf '  address      %s\n' "$DERIVED"
    fi
    printf '\n'
    if [ -n "$ADDRESS" ]; then
        printf 'The address was derived from the private key itself and compared with the one\n'
        printf 'you gave, so the key and the offset belong to the same seed.\n'
    elif [ -n "$DERIVED" ]; then
        printf 'The address was derived from the private key itself, not read back from the\n'
        printf 'miner. Compare it with the address 1miner printed, or pass --address to have\n'
        printf 'that compared here.\n'
    else
        printf 'The address could not be derived here, so nothing has confirmed that this key\n'
        printf 'belongs to the run that produced the offset. Import it and check the address\n'
        printf 'before sending anything to it.\n'
    fi
    printf '\n'
} >&2
