//! Hex parsing and EIP-55 formatting.

use crate::{Address, CoreError, Hash, Result, keccak256};

/// Decode a hex string, with or without a `0x`/`0X` prefix.
pub fn parse_hex(s: &str) -> Result<Vec<u8>> {
    let trimmed = s.trim();
    let body = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    if body.len() % 2 != 0 {
        return Err(CoreError::Parse(format!(
            "hex string has an odd number of digits: {s}"
        )));
    }
    hex::decode(body).map_err(|e| CoreError::Parse(format!("invalid hex {s}: {e}")))
}

fn parse_fixed<const N: usize>(s: &str, what: &str) -> Result<[u8; N]> {
    let bytes = parse_hex(s)?;
    if bytes.len() != N {
        return Err(CoreError::Parse(format!(
            "{what} must be {N} bytes, got {} in {s}",
            bytes.len()
        )));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes);
    Ok(out)
}

pub fn parse_address(s: &str) -> Result<Address> {
    parse_fixed::<20>(s, "address")
}

pub fn parse_hash(s: &str) -> Result<Hash> {
    parse_fixed::<32>(s, "32-byte value")
}

pub fn parse_magic(s: &str) -> Result<[u8; 16]> {
    parse_fixed::<16>(s, "16-byte magic")
}

/// EIP-55 mixed-case checksum encoding, with the `0x` prefix.
pub fn to_checksum_address(addr: &Address) -> String {
    let lower = hex::encode(addr);
    let digest = keccak256(lower.as_bytes());

    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in lower.chars().enumerate() {
        // Each hex digit is checksummed by the corresponding nibble of the hash.
        let nibble = if i % 2 == 0 {
            digest[i / 2] >> 4
        } else {
            digest[i / 2] & 0x0f
        };
        if c.is_ascii_digit() || nibble < 8 {
            out.push(c);
        } else {
            out.push(c.to_ascii_uppercase());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_matches_eip55_examples() {
        for expected in [
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
            "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
            "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
            "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        ] {
            let addr = parse_address(expected).unwrap();
            assert_eq!(to_checksum_address(&addr), expected);
        }
    }

    #[test]
    fn parses_with_and_without_prefix() {
        let a = parse_address("0x00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        let b = parse_address("00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_wrong_length_and_bad_digits() {
        assert!(parse_address("0xdeadbeef").is_err());
        assert!(parse_hash("0x00").is_err());
        assert!(parse_hex("0xzz").is_err());
        assert!(parse_hex("0x0").is_err());
    }

    #[test]
    fn empty_hex_is_allowed() {
        assert!(parse_hex("0x").unwrap().is_empty());
    }
}
