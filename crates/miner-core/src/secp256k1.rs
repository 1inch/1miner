//! Host-side secp256k1, used for profanity mode.
//!
//! This is not a constant-time implementation and must never touch a real
//! secret. It exists to build the GPU's precomputed generator table and to
//! verify results: given a seed public key and the offset the kernel reports,
//! recompute the address and check the miner told the truth.
//!
//! Field arithmetic is 4x u64 limbs, little-endian, reduced mod
//! `p = 2^256 - 2^32 - 977`.

use crate::{Address, CoreError, Result, address::eoa_address};

/// `p = 2^256 - C`, so a 512-bit product reduces by folding the high half
/// back in multiplied by C.
const P: [u64; 4] = [
    0xFFFF_FFFE_FFFF_FC2F,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];
const C: u64 = 0x1_0000_03D1; // 2^32 + 977

/// Order of the generator.
const N: [u64; 4] = [
    0xBFD2_5E8C_D036_4141,
    0xBAAE_DCE6_AF48_A03B,
    0xFFFF_FFFF_FFFF_FFFE,
    0xFFFF_FFFF_FFFF_FFFF,
];

const GX: [u64; 4] = [
    0x59F2_815B_16F8_1798,
    0x029B_FCDB_2DCE_28D9,
    0x55A0_6295_CE87_0B07,
    0x79BE_667E_F9DC_BBAC,
];
const GY: [u64; 4] = [
    0x9C47_D08F_FB10_D4B8,
    0xFD17_B448_A685_5419,
    0x5DA4_FBFC_0E11_08A8,
    0x483A_DA77_26A3_C465,
];

type Fe = [u64; 4];

fn is_zero(a: &Fe) -> bool {
    a.iter().all(|l| *l == 0)
}

fn cmp_ge(a: &Fe, b: &Fe) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn add_raw(a: &Fe, b: &Fe) -> (Fe, u64) {
    let mut out = [0u64; 4];
    let mut carry = 0u128;
    for i in 0..4 {
        let sum = a[i] as u128 + b[i] as u128 + carry;
        out[i] = sum as u64;
        carry = sum >> 64;
    }
    (out, carry as u64)
}

fn sub_raw(a: &Fe, b: &Fe) -> (Fe, u64) {
    let mut out = [0u64; 4];
    let mut borrow = 0i128;
    for i in 0..4 {
        let diff = a[i] as i128 - b[i] as i128 - borrow;
        out[i] = diff as u64;
        borrow = i128::from(diff < 0);
    }
    (out, borrow as u64)
}

fn add_mod(a: &Fe, b: &Fe, m: &Fe) -> Fe {
    let (sum, carry) = add_raw(a, b);
    if carry == 1 || cmp_ge(&sum, m) {
        sub_raw(&sum, m).0
    } else {
        sum
    }
}

fn sub_mod(a: &Fe, b: &Fe, m: &Fe) -> Fe {
    let (diff, borrow) = sub_raw(a, b);
    if borrow == 1 {
        add_raw(&diff, m).0
    } else {
        diff
    }
}

fn fe_add(a: &Fe, b: &Fe) -> Fe {
    add_mod(a, b, &P)
}

fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    sub_mod(a, b, &P)
}

/// Schoolbook 256x256 -> 512, then fold the high half back with `C`.
fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let mut wide = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let cur = wide[i + j] as u128 + (a[i] as u128) * (b[j] as u128) + carry;
            wide[i + j] = cur as u64;
            carry = cur >> 64;
        }
        wide[i + 4] = carry as u64;
    }
    reduce_wide(&wide)
}

fn reduce_wide(wide: &[u64; 8]) -> Fe {
    // value = lo + hi * 2^256, and 2^256 == C (mod p).
    let mut lo = [wide[0], wide[1], wide[2], wide[3]];
    let mut hi = [wide[4], wide[5], wide[6], wide[7]];

    while !is_zero(&hi) {
        // hi * C is at most 5 limbs.
        let mut prod = [0u64; 5];
        let mut carry = 0u128;
        for i in 0..4 {
            let cur = (hi[i] as u128) * (C as u128) + carry;
            prod[i] = cur as u64;
            carry = cur >> 64;
        }
        prod[4] = carry as u64;

        let (sum, carry) = add_raw(&lo, &[prod[0], prod[1], prod[2], prod[3]]);
        lo = sum;
        hi = [prod[4] + carry, 0, 0, 0];
    }

    while cmp_ge(&lo, &P) {
        lo = sub_raw(&lo, &P).0;
    }
    lo
}

fn fe_sqr(a: &Fe) -> Fe {
    fe_mul(a, a)
}

/// Inversion by Fermat's little theorem: `a^(p-2) mod p`. Slow but obviously
/// correct, and this only runs on the host.
fn fe_inv(a: &Fe) -> Fe {
    // p - 2
    let exp: Fe = [
        0xFFFF_FFFE_FFFF_FC2D,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
    ];
    let mut result: Fe = [1, 0, 0, 0];
    let mut base = *a;
    for limb in &exp {
        for bit in 0..64 {
            if (limb >> bit) & 1 == 1 {
                result = fe_mul(&result, &base);
            }
            base = fe_sqr(&base);
        }
    }
    result
}

/// Affine point; `None` represents the point at infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: Fe,
    pub y: Fe,
}

pub fn generator() -> Point {
    Point { x: GX, y: GY }
}

fn point_double(p: &Point) -> Option<Point> {
    if is_zero(&p.y) {
        return None;
    }
    // lambda = 3x^2 / 2y
    let three_x2 = {
        let x2 = fe_sqr(&p.x);
        fe_add(&fe_add(&x2, &x2), &x2)
    };
    let two_y = fe_add(&p.y, &p.y);
    let lambda = fe_mul(&three_x2, &fe_inv(&two_y));

    let x3 = fe_sub(&fe_sub(&fe_sqr(&lambda), &p.x), &p.x);
    let y3 = fe_sub(&fe_mul(&lambda, &fe_sub(&p.x, &x3)), &p.y);
    Some(Point { x: x3, y: y3 })
}

pub fn point_add(a: Option<&Point>, b: Option<&Point>) -> Option<Point> {
    let (a, b) = match (a, b) {
        (None, None) => return None,
        (Some(p), None) | (None, Some(p)) => return Some(*p),
        (Some(a), Some(b)) => (a, b),
    };

    if a.x == b.x {
        return if a.y == b.y { point_double(a) } else { None };
    }

    let lambda = fe_mul(&fe_sub(&b.y, &a.y), &fe_inv(&fe_sub(&b.x, &a.x)));
    let x3 = fe_sub(&fe_sub(&fe_sqr(&lambda), &a.x), &b.x);
    let y3 = fe_sub(&fe_mul(&lambda, &fe_sub(&a.x, &x3)), &a.y);
    Some(Point { x: x3, y: y3 })
}

/// Scalar multiplication by double-and-add over a big-endian 32-byte scalar.
pub fn scalar_mul(scalar: &[u8; 32], point: &Point) -> Option<Point> {
    let mut acc: Option<Point> = None;
    let mut addend = *point;
    // Walk least-significant bit first.
    for byte in scalar.iter().rev() {
        for bit in 0..8 {
            if (byte >> bit) & 1 == 1 {
                acc = point_add(acc.as_ref(), Some(&addend));
            }
            match point_double(&addend) {
                Some(d) => addend = d,
                None => return acc,
            }
        }
    }
    acc
}

pub fn scalar_mul_generator(scalar: &[u8; 32]) -> Option<Point> {
    scalar_mul(scalar, &generator())
}

fn fe_to_be_bytes(a: &Fe) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[24 - i * 8..32 - i * 8].copy_from_slice(&a[i].to_be_bytes());
    }
    out
}

fn fe_from_be_bytes(b: &[u8; 32]) -> Fe {
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&b[24 - i * 8..32 - i * 8]);
        out[i] = u64::from_be_bytes(limb);
    }
    out
}

impl Point {
    pub fn to_bytes(self) -> ([u8; 32], [u8; 32]) {
        (fe_to_be_bytes(&self.x), fe_to_be_bytes(&self.y))
    }

    pub fn address(&self) -> Address {
        let (x, y) = self.to_bytes();
        eoa_address(&x, &y)
    }

    /// Reject anything not on `y^2 = x^3 + 7`, so a mistyped public key fails
    /// immediately rather than after hours of mining.
    pub fn is_on_curve(&self) -> bool {
        let lhs = fe_sqr(&self.y);
        let rhs = fe_add(&fe_mul(&fe_sqr(&self.x), &self.x), &[7, 0, 0, 0]);
        lhs == rhs
    }
}

/// Parse the 128-hex-character uncompressed public key profanity accepts (no
/// `0x04` prefix) and check it lies on the curve.
pub fn parse_public_key(s: &str) -> Result<Point> {
    let bytes = crate::hexutil::parse_hex(s)?;
    let bytes = match bytes.len() {
        64 => bytes,
        65 if bytes[0] == 0x04 => bytes[1..].to_vec(),
        other => {
            return Err(CoreError::Parse(format!(
                "public key must be 128 hex characters (64 bytes, no 0x04 prefix), got {other} bytes"
            )));
        }
    };

    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&bytes[..32]);
    y.copy_from_slice(&bytes[32..]);
    let point = Point {
        x: fe_from_be_bytes(&x),
        y: fe_from_be_bytes(&y),
    };
    if !point.is_on_curve() {
        return Err(CoreError::Parse(
            "public key is not a point on secp256k1".into(),
        ));
    }
    Ok(point)
}

/// `(a + b) mod n`, for combining a seed private key with a mined offset.
pub fn add_scalars_mod_n(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let sum = add_mod(&fe_from_be_bytes(a), &fe_from_be_bytes(b), &N);
    fe_to_be_bytes(&sum)
}

/// The address reached by walking `seed_pub` forward by `offset * G`. This is
/// the check that turns a reported profanity hit into a verified one.
pub fn address_for_offset(seed_pub: &Point, offset: &[u8; 32]) -> Option<Address> {
    let delta = scalar_mul_generator(offset);
    point_add(Some(seed_pub), delta.as_ref()).map(|p| p.address())
}

/// The generator table the profanity kernel indexes: for each of the 32 salt
/// bytes and each non-zero byte value, `value * 256^byte_index * G`.
/// Layout is `[byte_index * 255 + (value - 1)]`, matching precomp.cpp.
pub fn generator_table() -> Vec<Point> {
    let mut table = Vec::with_capacity(32 * 255);
    let mut base = generator();
    for _ in 0..32 {
        let mut acc = base;
        for _ in 0..255 {
            table.push(acc);
            acc = point_add(Some(&acc), Some(&base)).expect("no inverse pair within a byte lane");
        }
        // Advance the lane by a factor of 256.
        for _ in 0..8 {
            base = point_double(&base).expect("doubling the generator never hits infinity");
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexutil::parse_address;

    fn scalar(v: u64) -> [u8; 32] {
        let mut s = [0u8; 32];
        s[24..].copy_from_slice(&v.to_be_bytes());
        s
    }

    #[test]
    fn generator_is_on_curve() {
        assert!(generator().is_on_curve());
    }

    #[test]
    fn known_small_multiples() {
        // 2G and 3G, from the standard secp256k1 test vectors.
        let two_g = scalar_mul_generator(&scalar(2)).unwrap();
        let (x, _) = two_g.to_bytes();
        assert_eq!(
            hex::encode(x),
            "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"
        );

        let three_g = scalar_mul_generator(&scalar(3)).unwrap();
        let (x, _) = three_g.to_bytes();
        assert_eq!(
            hex::encode(x),
            "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"
        );
    }

    #[test]
    fn addition_agrees_with_scalar_mul() {
        let a = scalar_mul_generator(&scalar(7)).unwrap();
        let b = scalar_mul_generator(&scalar(11)).unwrap();
        let sum = point_add(Some(&a), Some(&b)).unwrap();
        assert_eq!(sum, scalar_mul_generator(&scalar(18)).unwrap());
    }

    #[test]
    fn a_plus_negative_a_is_infinity() {
        let a = scalar_mul_generator(&scalar(5)).unwrap();
        let neg = Point {
            x: a.x,
            y: fe_sub(&[0, 0, 0, 0], &a.y),
        };
        assert!(point_add(Some(&a), Some(&neg)).is_none());
    }

    /// Private key 1 corresponds to the generator, whose address is well known.
    #[test]
    fn address_of_generator() {
        assert_eq!(
            generator().address(),
            parse_address("0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf").unwrap()
        );
    }

    #[test]
    fn public_key_parsing_rejects_off_curve() {
        let (x, y) = generator().to_bytes();
        let good = format!("{}{}", hex::encode(x), hex::encode(y));
        assert!(parse_public_key(&good).is_ok());
        // Also accepts the 0x04-prefixed form.
        assert!(parse_public_key(&format!("04{good}")).is_ok());

        let bad = format!("{}{}", hex::encode(x), hex::encode([0xAAu8; 32]));
        assert!(parse_public_key(&bad).is_err());
        assert!(parse_public_key("deadbeef").is_err());
    }

    /// The offset round-trip that `--verify` relies on: mining reports `k`, and
    /// `seed_pub + k*G` must equal the address derived from `seed_priv + k`.
    #[test]
    fn offset_round_trip() {
        let seed_priv = scalar(1234567);
        let seed_pub = scalar_mul_generator(&seed_priv).unwrap();
        let offset = scalar(98765);

        let via_points = address_for_offset(&seed_pub, &offset).unwrap();
        let combined = add_scalars_mod_n(&seed_priv, &offset);
        let via_scalar = scalar_mul_generator(&combined).unwrap().address();

        assert_eq!(via_points, via_scalar);
    }

    #[test]
    fn scalar_addition_wraps_at_the_order() {
        let n_minus_one = {
            let mut s = fe_to_be_bytes(&N);
            s[31] -= 1;
            s
        };
        assert_eq!(add_scalars_mod_n(&n_minus_one, &scalar(1)), [0u8; 32]);
        assert_eq!(add_scalars_mod_n(&n_minus_one, &scalar(2)), scalar(1));
    }

    /// Validate the whole table against profanity2's checked-in precomp.cpp.
    /// This pins both the field arithmetic and the index layout at once, and is
    /// the reason the table can be generated rather than vendored.
    ///
    /// Skipped when the reference checkout is absent (Docker, packaged builds).
    #[test]
    fn generator_table_matches_profanity2_precomp() {
        let path = "../../references/profanity2/precomp.cpp";
        let Ok(source) = std::fs::read_to_string(path) else {
            eprintln!("skipping: {path} not present");
            return;
        };

        // Each point is 16 little-endian u32 literals: 8 for x, then 8 for y.
        let words: Vec<u32> = source
            .split("0x")
            .skip(1)
            .filter_map(|tok| {
                let digits: String = tok.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                u32::from_str_radix(&digits, 16).ok()
            })
            .collect();
        assert_eq!(words.len(), 8160 * 16, "unexpected precomp.cpp layout");

        let to_fe = |w: &[u32]| -> Fe {
            [
                w[0] as u64 | ((w[1] as u64) << 32),
                w[2] as u64 | ((w[3] as u64) << 32),
                w[4] as u64 | ((w[5] as u64) << 32),
                w[6] as u64 | ((w[7] as u64) << 32),
            ]
        };

        let table = generator_table();
        for (i, point) in table.iter().enumerate() {
            let base = i * 16;
            assert_eq!(
                point.x,
                to_fe(&words[base..base + 8]),
                "x mismatch at index {i}"
            );
            assert_eq!(
                point.y,
                to_fe(&words[base + 8..base + 16]),
                "y mismatch at index {i}"
            );
        }
    }

    /// The same table, pinned in a way that cannot skip. `references/` is
    /// gitignored, so the cross-check above is vacuous on a fresh clone and in
    /// every container build — which is everywhere except a developer's own
    /// machine. This digest was recorded here, where that cross-check passes, so
    /// it inherits its authority; what it adds is regression detection that runs
    /// unconditionally.
    ///
    /// A digest cannot replace the cross-check: it pins the table against
    /// itself, not against profanity2. Recompute it only after the cross-check
    /// has passed against the reference tree.
    #[test]
    fn generator_table_digest_is_pinned() {
        let mut bytes = Vec::with_capacity(8160 * 64);
        for point in generator_table() {
            let (x, y) = point.to_bytes();
            bytes.extend_from_slice(&x);
            bytes.extend_from_slice(&y);
        }
        assert_eq!(
            hex::encode(crate::keccak256(&bytes)),
            "4003072db7dc32261687856a7630945f1ec81babd0468846d6d022ebe8ed74ee"
        );
    }

    #[test]
    fn generator_table_shape_and_first_entries() {
        let table = generator_table();
        assert_eq!(table.len(), 8160);
        // Lane 0 holds 1G, 2G, 3G ...
        assert_eq!(table[0], generator());
        assert_eq!(table[1], scalar_mul_generator(&scalar(2)).unwrap());
        assert_eq!(table[254], scalar_mul_generator(&scalar(255)).unwrap());
        // Lane 1 starts at 256G.
        assert_eq!(table[255], scalar_mul_generator(&scalar(256)).unwrap());
        assert_eq!(table[256], scalar_mul_generator(&scalar(512)).unwrap());
    }
}
