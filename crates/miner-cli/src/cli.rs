//! Command line surface.
//!
//! Modes are subcommands so that clap can enforce each mode's own required
//! inputs. Long flags are authoritative: the upstream miners disagree about
//! what `-A` means (deployer in ERADICATE2, caller in ERADICATE3), so short
//! aliases are only offered where they mean the same thing everywhere.

use clap::{Args, Parser, Subcommand};
use miner_backend::{KeccakVariant, Tuning};
use miner_core::{
    DEFAULT_PROXY_CODE_HASH, Hash, MineMode, ScoreSpec, keccak256, parse_address, parse_hash,
    parse_hex,
};

#[derive(Parser, Debug)]
#[command(
    name = "1miner",
    version,
    about = "GPU miner for vanity Ethereum addresses: profanity, create2, create3 and 1inch Address NFT",
    long_about = None,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Vanity account or contract address, searching private-key offsets.
    Profanity(ProfanityArgs),
    /// Vanity CREATE2 address, searching salts for a deployer and init code.
    Create2(Create2Args),
    /// Vanity CREATE3 address, searching salts for a factory.
    Create3(Create3Args),
    /// 1inch Address NFT: search the bytes16 magic for mint()/mintFor().
    #[command(name = "1nft")]
    Nft(NftArgs),
    /// Verify the selected backend against known vectors and the CPU reference.
    SelfTest(SelfTestArgs),
}

#[derive(Args, Debug)]
pub struct ProfanityArgs {
    /// Seed public key: 128 hex characters, no 0x04 prefix.
    ///
    /// Only the public key is ever accepted. The miner reports an offset to add
    /// to your seed private key, so the key itself never leaves your machine
    /// and mining can safely run on rented hardware.
    #[arg(long = "public-key", short = 'z')]
    pub public_key: String,

    /// Score the contract this key would deploy at nonce 0, not the account.
    #[arg(long, short = 'c')]
    pub contract: bool,

    #[command(flatten)]
    pub scoring: ScoringArgs,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Args, Debug)]
pub struct Create2Args {
    /// Contract performing the CREATE2.
    #[arg(long)]
    pub deployer: Option<String>,

    /// Init code as hex.
    #[arg(long = "init-code")]
    pub init_code: Option<String>,
    /// File containing the init code as hex.
    #[arg(long = "init-code-file")]
    pub init_code_file: Option<String>,
    /// keccak256 of the init code, if you already have it.
    #[arg(long = "init-code-hash")]
    pub init_code_hash: Option<String>,

    #[command(flatten)]
    pub scoring: ScoringArgs,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Args, Debug)]
pub struct Create3Args {
    /// CREATE3 factory whose address the result depends on.
    #[arg(long)]
    pub deployer: Option<String>,

    /// keccak256 of the factory's CREATE2 proxy bytecode.
    #[arg(long = "bytecode-hash")]
    pub bytecode_hash: Option<String>,

    #[command(flatten)]
    pub scoring: ScoringArgs,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Args, Debug)]
pub struct NftArgs {
    /// Deployer contract. Required: there is deliberately no default, because
    /// mining a magic against an assumed deployer yields an unusable result.
    #[arg(long)]
    pub deployer: Option<String>,

    /// Account the vanity address will be minted for, and which will own the
    /// resulting token.
    // `--caller-address` stays as an undocumented alias, because that is what
    // the tool this mode came from called it.
    #[arg(long = "mint-for", alias = "caller-address")]
    pub mint_for: Option<String>,

    /// keccak256 of the deployer's CREATE2 proxy bytecode.
    #[arg(long = "bytecode-hash")]
    pub bytecode_hash: Option<String>,

    #[command(flatten)]
    pub scoring: ScoringArgs,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Args, Debug)]
pub struct SelfTestArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Exactly one scoring mode must be chosen.
#[derive(Args, Debug, Default)]
pub struct ScoringArgs {
    /// Measure throughput without scoring anything.
    #[arg(long)]
    pub benchmark: bool,
    /// Count zero nibbles anywhere in the address.
    #[arg(long)]
    pub zeros: bool,
    /// Count a-f nibbles anywhere.
    #[arg(long)]
    pub letters: bool,
    /// Count 0-9 nibbles anywhere.
    #[arg(long)]
    pub numbers: bool,
    /// Score palindromes measured out from the centre.
    #[arg(long)]
    pub mirror: bool,
    /// Score leading bytes whose two nibbles match (00, 11, ... ff).
    #[arg(long = "leading-doubles")]
    pub leading_doubles: bool,
    /// Count whole zero bytes, which makes calldata cheaper.
    #[arg(long = "zero-bytes")]
    pub zero_bytes: bool,
    /// Score a leading run of one hex digit.
    #[arg(long)]
    pub leading: Option<char>,
    /// Score a mask anchored at the start; non-hex characters are wildcards.
    #[arg(long)]
    pub matching: Option<String>,
    /// Score a mask anchored at the end.
    #[arg(long)]
    pub trailing: Option<String>,
    /// Report every address matching the mask in full, rather than climbing
    /// towards a best score. Non-hex characters are wildcards.
    #[arg(long, short = 'e')]
    pub exact: Option<String>,
    /// Score a leading run of nibbles within --min..=--max.
    #[arg(long = "leading-range")]
    pub leading_range: bool,
    /// Count nibbles within --min..=--max anywhere.
    #[arg(long)]
    pub range: bool,

    // Deliberately Option rather than a defaulted u8: only --range and
    // --leading-range read these, and a clap default would make "supplied"
    // indistinguishable from "defaulted", which is how they came to be accepted
    // and silently ignored beside every other scoring mode.
    /// Range minimum nibble, 0-15. Defaults to 0.
    #[arg(long, short = 'm')]
    pub min: Option<u8>,
    /// Range maximum nibble, 0-15. Defaults to 15.
    #[arg(long, short = 'M')]
    pub max: Option<u8>,
}

/// Highest `--inverse-size` accepted. `PROFANITY_INVERSE_SIZE` is the length of
/// two private `mp_number` arrays in profanity.cl, 32 bytes each, so 1024 asks
/// one work item for 64 KB of private memory — already past any real device.
/// Above that the build spills or fails to compile with no useful diagnostic.
const MAX_INVERSE_SIZE: usize = 1024;

/// Zero reaches the backends as a division or an empty round: `--inverse-size 0`
/// divides by zero at `size / job.tuning.inverse_size` in `run_profanity`, and
/// `--size 0` leaves the salt loop with nothing to enqueue, so it spins at full
/// speed reporting a rate of zero.
fn positive(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("must be at least 1".into()),
        Ok(v) => Ok(v),
        Err(e) => Err(e.to_string()),
    }
}

fn inverse_size(s: &str) -> Result<usize, String> {
    match positive(s)? {
        v if v > MAX_INVERSE_SIZE => Err(format!("must be at most {MAX_INVERSE_SIZE}")),
        v => Ok(v),
    }
}

#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Compute backend.
    #[arg(long, default_value = "opencl", value_parser = ["opencl", "metal", "cpu"])]
    pub backend: String,

    /// Kernel variant, so implementations can be raced against each other.
    #[arg(long, default_value = "tuned", value_parser = ["tuned", "plain"])]
    pub kernel: String,

    /// OpenCL local work size; 0 lets the driver decide.
    #[arg(long, short = 'w', default_value_t = 128)]
    pub work: usize,

    /// Largest single enqueue.
    #[arg(long = "work-max", short = 'W', value_parser = positive)]
    pub work_max: Option<usize>,

    /// Salt candidates per round per device.
    #[arg(long, short = 'S', default_value_t = 16_777_216, value_parser = positive)]
    pub size: usize,

    /// profanity batched-inverse width.
    #[arg(long = "inverse-size", short = 'i', default_value_t = 255, value_parser = inverse_size)]
    pub inverse_size: usize,

    /// profanity parallel inverse batches.
    #[arg(long = "inverse-multiple", short = 'I', default_value_t = 16_384, value_parser = positive)]
    pub inverse_multiple: usize,

    /// Skip the device at this index. Repeatable.
    #[arg(long, short = 's')]
    pub skip: Vec<usize>,

    /// Ignore any cached compiled kernel.
    #[arg(long = "no-cache", short = 'n')]
    pub no_cache: bool,

    /// Do not re-derive each hit on the CPU before printing it.
    #[arg(long = "no-verify")]
    pub no_verify: bool,

    /// Stop after this many seconds.
    #[arg(long)]
    pub seconds: Option<u64>,

    /// Exclude this many seconds from the measured summary, so a benchmark
    /// figure is not diluted by kernel compilation and the first slow round.
    #[arg(long, default_value_t = 0)]
    pub warmup: u64,

    /// Number of CPU worker threads, for --backend cpu.
    #[arg(long)]
    pub threads: Option<usize>,
}

impl CommonArgs {
    pub fn tuning(&self) -> Tuning {
        Tuning {
            work_size: self.work,
            work_max: self.work_max,
            round_size: self.size,
            inverse_size: self.inverse_size,
            inverse_multiple: self.inverse_multiple,
            skip_devices: self.skip.clone(),
            no_cache: self.no_cache,
            warmup: std::time::Duration::from_secs(self.warmup),
        }
    }

    pub fn keccak(&self) -> KeccakVariant {
        KeccakVariant::parse(&self.kernel).unwrap_or_default()
    }
}

/// A resolved scoring choice: the specification the kernels take, plus the
/// all-or-nothing threshold when `--exact` was used.
#[derive(Debug, Clone)]
pub struct Scoring {
    pub spec: ScoreSpec,
    pub exact_score: Option<u32>,
}

impl ScoringArgs {
    /// Resolve to exactly one scoring specification, or explain what is wrong.
    pub fn resolve(&self) -> anyhow::Result<Scoring> {
        let mut chosen: Vec<(&str, ScoreSpec)> = Vec::new();

        if self.benchmark {
            chosen.push(("--benchmark", ScoreSpec::benchmark()));
        }
        if self.zeros {
            chosen.push(("--zeros", ScoreSpec::zeros()));
        }
        if self.letters {
            chosen.push(("--letters", ScoreSpec::letters()));
        }
        if self.numbers {
            chosen.push(("--numbers", ScoreSpec::numbers()));
        }
        if self.mirror {
            chosen.push(("--mirror", ScoreSpec::mirror()));
        }
        if self.leading_doubles {
            chosen.push(("--leading-doubles", ScoreSpec::doubles()));
        }
        if self.zero_bytes {
            chosen.push(("--zero-bytes", ScoreSpec::zero_bytes()));
        }
        if let Some(c) = self.leading {
            chosen.push(("--leading", ScoreSpec::leading(c)?));
        }
        if let Some(p) = &self.matching {
            chosen.push(("--matching", ScoreSpec::matching(p)?));
        }
        if let Some(p) = &self.trailing {
            chosen.push(("--trailing", ScoreSpec::trailing(p)?));
        }
        if let Some(p) = &self.exact {
            chosen.push(("--exact", ScoreSpec::matching(p)?));
        }
        let min = self.min.unwrap_or(0);
        let max = self.max.unwrap_or(15);
        if self.leading_range {
            chosen.push(("--leading-range", ScoreSpec::leading_range(min, max)?));
        }
        if self.range {
            chosen.push(("--range", ScoreSpec::range(min, max)?));
        }

        match chosen.len() {
            1 => {
                let (name, spec) = chosen.pop().expect("length checked");

                // Checked here rather than before the count, so that a bound
                // with no scoring mode at all still gets the more useful
                // "choose a scoring mode" below.
                if !(self.range || self.leading_range) {
                    let bounds: Vec<&str> = [self.min.map(|_| "--min"), self.max.map(|_| "--max")]
                        .into_iter()
                        .flatten()
                        .collect();
                    if !bounds.is_empty() {
                        // Phrased without a pronoun so it reads for one bound
                        // or both.
                        anyhow::bail!(
                            "{name} does not read {}; use --range or --leading-range to bound \
                             the nibbles",
                            bounds.join(" and ")
                        );
                    }
                }

                let exact_score = if name == "--exact" {
                    let needed = spec.constrained_bytes();
                    if needed == 0 {
                        anyhow::bail!(
                            "--exact needs at least one hex digit to match against; \
                             a mask of only wildcards matches everything"
                        );
                    }
                    Some(needed)
                } else {
                    None
                };
                Ok(Scoring { spec, exact_score })
            }
            0 => anyhow::bail!(
                "choose a scoring mode, for example --leading 0, --matching dead, --zeros or --benchmark"
            ),
            _ => {
                let names: Vec<&str> = chosen.iter().map(|(n, _)| *n).collect();
                anyhow::bail!("choose one scoring mode, got {}", names.join(" and "))
            }
        }
    }

    pub fn is_benchmark(&self) -> bool {
        self.benchmark
    }
}

/// Resolve the deployer, allowing it to be omitted only for a benchmark, where
/// the output is a throughput figure rather than a usable result.
pub fn resolve_deployer(
    deployer: Option<&String>,
    mode: MineMode,
    benchmark: bool,
) -> anyhow::Result<[u8; 20]> {
    match deployer {
        Some(s) => Ok(parse_address(s)?),
        None if benchmark => Ok([0u8; 20]),
        None => anyhow::bail!(
            "{} requires --deployer; it is never assumed, because a salt mined against \
             the wrong deployer produces an address that looks valid and is unusable",
            mode.as_str()
        ),
    }
}

pub fn resolve_bytecode_hash(value: Option<&String>) -> anyhow::Result<Hash> {
    match value {
        Some(s) => Ok(parse_hash(s)?),
        None => Ok(DEFAULT_PROXY_CODE_HASH),
    }
}

/// CREATE2 needs `keccak256(initCode)`, from whichever form was supplied.
pub fn resolve_init_code_hash(
    init_code: Option<&String>,
    init_code_file: Option<&String>,
    init_code_hash: Option<&String>,
    benchmark: bool,
) -> anyhow::Result<Hash> {
    let supplied = [
        init_code.map(|_| "--init-code"),
        init_code_file.map(|_| "--init-code-file"),
        init_code_hash.map(|_| "--init-code-hash"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    if supplied.len() > 1 {
        anyhow::bail!("give only one of {}", supplied.join(", "));
    }

    if let Some(h) = init_code_hash {
        return Ok(parse_hash(h)?);
    }
    if let Some(code) = init_code {
        return Ok(keccak256(&parse_hex(code)?));
    }
    if let Some(path) = init_code_file {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read init code from {path}: {e}"))?;
        return Ok(keccak256(&parse_hex(text.trim())?));
    }
    if benchmark {
        return Ok(keccak256(&[]));
    }
    anyhow::bail!("create2 requires --init-code, --init-code-file or --init-code-hash")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn positive_rejects_zero_and_inverse_size_is_bounded() {
        assert!(positive("0").is_err());
        assert_eq!(positive("1").unwrap(), 1);
        assert!(positive("-1").is_err());

        assert!(inverse_size("0").is_err());
        assert_eq!(inverse_size("1").unwrap(), 1);
        assert_eq!(
            inverse_size(&MAX_INVERSE_SIZE.to_string()).unwrap(),
            MAX_INVERSE_SIZE
        );
        assert!(inverse_size(&(MAX_INVERSE_SIZE + 1).to_string()).is_err());
    }

    /// Wiring a parser onto three flags and forgetting the fourth is the way
    /// this fix fails, so every flag is checked through clap rather than the
    /// parsers being trusted on their own.
    #[test]
    fn degenerate_tuning_values_are_rejected_on_every_flag() {
        let parse = |flag: &str, value: &str| {
            Cli::try_parse_from(["1miner", "create3", "--zeros", flag, value])
        };

        for flag in [
            "--size",
            "--inverse-size",
            "--inverse-multiple",
            "--work-max",
        ] {
            assert!(parse(flag, "0").is_err(), "{flag} 0 should be rejected");
            assert!(parse(flag, "1").is_ok(), "{flag} 1 should be accepted");
        }

        assert!(parse("--inverse-size", &(MAX_INVERSE_SIZE + 1).to_string()).is_err());
        // 0 stays meaningful here: it lets the driver pick the local work size.
        assert!(parse("--work", "0").is_ok());
    }

    #[test]
    fn scoring_requires_exactly_one_mode() {
        let none = ScoringArgs::default();
        assert!(none.resolve().is_err());

        let one = ScoringArgs {
            zeros: true,
            ..Default::default()
        };
        assert!(one.resolve().is_ok());

        let two = ScoringArgs {
            zeros: true,
            letters: true,
            ..Default::default()
        };
        let err = two.resolve().unwrap_err().to_string();
        assert!(
            err.contains("--zeros") && err.contains("--letters"),
            "{err}"
        );
    }

    /// A bound that does nothing used to be accepted in silence, which reads as
    /// a narrower search than the one actually running.
    #[test]
    fn range_bounds_are_rejected_beside_a_mode_that_ignores_them() {
        let ignored = ScoringArgs {
            zeros: true,
            min: Some(5),
            ..Default::default()
        };
        let err = ignored.resolve().unwrap_err().to_string();
        assert!(err.contains("--min") && err.contains("--zeros"), "{err}");

        let both = ScoringArgs {
            leading_doubles: true,
            min: Some(3),
            max: Some(9),
            ..Default::default()
        };
        let err = both.resolve().unwrap_err().to_string();
        assert!(err.contains("--min and --max"), "{err}");

        // No scoring mode at all is the more useful complaint of the two.
        let only_bounds = ScoringArgs {
            max: Some(9),
            ..Default::default()
        };
        assert!(
            only_bounds
                .resolve()
                .unwrap_err()
                .to_string()
                .contains("choose a scoring mode")
        );

        let bounded = ScoringArgs {
            range: true,
            min: Some(5),
            max: Some(9),
            ..Default::default()
        };
        let spec = bounded.resolve().unwrap().spec;
        assert_eq!((spec.data1[0], spec.data2[0]), (5, 9));

        // Omitted bounds still mean the whole nibble range.
        let unbounded = ScoringArgs {
            leading_range: true,
            ..Default::default()
        };
        let spec = unbounded.resolve().unwrap().spec;
        assert_eq!((spec.data1[0], spec.data2[0]), (0, 15));
    }

    #[test]
    fn deployer_is_required_unless_benchmarking() {
        assert!(resolve_deployer(None, MineMode::Nft, false).is_err());
        assert_eq!(
            resolve_deployer(None, MineMode::Nft, true).unwrap(),
            [0u8; 20]
        );
        let addr = "0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf".to_string();
        assert!(resolve_deployer(Some(&addr), MineMode::Create3, false).is_ok());
    }

    #[test]
    fn init_code_forms_agree_and_conflict_is_rejected() {
        let code = "0xdeadbeef".to_string();
        let by_code = resolve_init_code_hash(Some(&code), None, None, false).unwrap();
        let hash = hex::encode(by_code);
        let by_hash = resolve_init_code_hash(None, None, Some(&hash), false).unwrap();
        assert_eq!(by_code, by_hash);

        assert!(resolve_init_code_hash(Some(&code), None, Some(&hash), false).is_err());
        assert!(resolve_init_code_hash(None, None, None, false).is_err());
        assert!(resolve_init_code_hash(None, None, None, true).is_ok());
    }

    #[test]
    fn bytecode_hash_defaults_to_the_standard_proxy() {
        assert_eq!(
            resolve_bytecode_hash(None).unwrap(),
            DEFAULT_PROXY_CODE_HASH
        );
    }
}
