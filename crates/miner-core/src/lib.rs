//! Shared, backend-agnostic core for the `1miner` GPU address miner.
//!
//! Everything here runs on the CPU and is the reference against which the
//! OpenCL, Metal and NEON kernels are checked. Keeping the derivations in one
//! place is deliberate: the same address maths is reimplemented several times
//! for different accelerators, and every mistake in that maths produces a
//! well-formed but wrong address rather than an obvious failure.

pub mod address;
pub mod hexutil;
pub mod mode;
pub mod scoring;
pub mod secp256k1;

pub use address::{
    Address, DEFAULT_PROXY_CODE_HASH, Hash, ONEINCH_NFT_DEPLOYER, PROXY_CHILD_BYTECODE, Salt,
    create2_address, create2_preimage, create3_address, create_address, eoa_address, nft_salt,
};
pub use hexutil::{parse_address, parse_hash, parse_hex, parse_magic, to_checksum_address};
pub use mode::{MineMode, ModeConfig, ProfanityConfig, SaltConfig};
pub use scoring::{ScoreFn, ScoreSpec, score};

use sha3::{Digest, Keccak256};

/// Keccak-256 as used by Ethereum (not the NIST SHA3-256 padding).
pub fn keccak256(data: &[u8]) -> Hash {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Keccak256::digest(data));
    out
}

/// Errors surfaced by parsing and configuration in this crate.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("{0}")]
    Parse(String),
    #[error("{0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
