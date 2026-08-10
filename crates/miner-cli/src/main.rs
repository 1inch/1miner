mod cli;
mod report;
mod selftest;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use miner_backend::{Backend, EXACT_CAPACITY, Job, Tuning};
use miner_core::{Address, Hash, MineMode, ModeConfig, ProfanityConfig, Salt, SaltConfig};
use rand::RngCore;

use cli::{Cli, Command, CommonArgs, Scoring, ScoringArgs};
use report::TerminalReporter;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let stop = Arc::new(AtomicBool::new(false));
    install_interrupt_handler(Arc::clone(&stop));

    match cli.command {
        Command::SelfTest(args) => selftest::run(&args.common),
        Command::Profanity(args) => {
            let score = args.scoring.resolve()?;
            let seed = miner_core::secp256k1::parse_public_key(&args.public_key)
                .context("invalid --public-key")?;
            let mode = ModeConfig::Profanity(ProfanityConfig {
                seed_public_key: seed,
                contract: args.contract,
            });
            mine(mode, score, &args.common, &args.scoring, stop)
        }
        Command::Create2(args) => {
            let score = args.scoring.resolve()?;
            let benchmark = args.scoring.is_benchmark();
            let deployer =
                cli::resolve_deployer(args.deployer.as_ref(), MineMode::Create2, benchmark)?;
            let code_hash = cli::resolve_init_code_hash(
                args.init_code.as_ref(),
                args.init_code_file.as_ref(),
                args.init_code_hash.as_ref(),
                benchmark,
            )?;
            let mode = salt_mode(MineMode::Create2, deployer, code_hash, None)?;
            mine(mode, score, &args.common, &args.scoring, stop)
        }
        Command::Create3(args) => {
            let score = args.scoring.resolve()?;
            let benchmark = args.scoring.is_benchmark();
            let deployer =
                cli::resolve_deployer(args.deployer.as_ref(), MineMode::Create3, benchmark)?;
            let code_hash = cli::resolve_bytecode_hash(args.bytecode_hash.as_ref())?;
            let mode = salt_mode(MineMode::Create3, deployer, code_hash, None)?;
            mine(mode, score, &args.common, &args.scoring, stop)
        }
        Command::Nft(args) => {
            let score = args.scoring.resolve()?;
            let benchmark = args.scoring.is_benchmark();
            let deployer = cli::resolve_deployer(args.deployer.as_ref(), MineMode::Nft, benchmark)?;
            let caller = match args.mint_for.as_ref() {
                Some(s) => miner_core::parse_address(s)?,
                None if benchmark => [0u8; 20],
                None => anyhow::bail!(
                    "1nft requires --mint-for, the account the vanity address is minted for"
                ),
            };
            let code_hash = cli::resolve_bytecode_hash(args.bytecode_hash.as_ref())?;
            let mode = salt_mode(MineMode::Nft, deployer, code_hash, Some(caller))?;
            mine(mode, score, &args.common, &args.scoring, stop)
        }
    }
}

fn random_salt() -> Salt {
    let mut salt = [0u8; 32];
    rand::rng().fill_bytes(&mut salt);
    salt
}

fn salt_mode(
    mode: MineMode,
    deployer: Address,
    code_hash: Hash,
    caller: Option<Address>,
) -> anyhow::Result<ModeConfig> {
    let cfg = SaltConfig::new(mode, deployer, code_hash, random_salt(), caller)?;
    Ok(ModeConfig::Salt(cfg))
}

fn install_interrupt_handler(stop: Arc<AtomicBool>) {
    let handler = move || {
        // Second interrupt exits immediately, in case a device is wedged.
        if stop.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!("\ninterrupted, finishing the current round...");
    };
    if let Err(e) = ctrlc::set_handler(handler) {
        eprintln!("warning: could not install interrupt handler: {e}");
    }
}

fn mine(
    mode: ModeConfig,
    scoring: Scoring,
    common: &CommonArgs,
    args: &ScoringArgs,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let tuning = common.tuning();
    let (labels, masks): (Vec<String>, Vec<_>) = scoring
        .exact
        .clone()
        .unwrap_or_default()
        .into_iter()
        .unzip();
    let job = Job {
        mode: mode.clone(),
        score: scoring.spec,
        keccak: common.keccak(),
        tuning: tuning.clone(),
        duration: common.seconds.map(Duration::from_secs),
        verify: !common.no_verify,
        exact: scoring.exact.map(|_| masks),
    };

    let mut backend = open_backend(common, &mode, &tuning)?;
    print_devices(backend.as_ref(), common, &mode, args);
    if !labels.is_empty() {
        let plural = if labels.len() == 1 { "mask" } else { "masks" };
        println!(
            "Exact: reporting every address matching {} {plural} in full, \
             up to {EXACT_CAPACITY} per round per device.",
            labels.len()
        );
        println!();
    }

    let mut reporter = TerminalReporter::new(mode.mode(), false, labels);
    let should_stop = move || stop.load(Ordering::SeqCst);
    backend.run(&job, &mut reporter, &should_stop)?;

    println!();
    if reporter.unverified > 0 {
        anyhow::bail!(
            "{} of {} reported hits failed CPU re-derivation; treat the results as unsafe \
             and report this, since it means the kernel and the reference disagree",
            reporter.unverified,
            reporter.hits
        );
    }
    Ok(())
}

fn open_backend(
    common: &CommonArgs,
    mode: &ModeConfig,
    tuning: &Tuning,
) -> anyhow::Result<Box<dyn Backend>> {
    match common.backend.as_str() {
        "cpu" => Ok(Box::new(miner_backend::cpu::CpuBackend::new(
            common.threads,
        ))),
        "metal" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if !miner_backend::metal::supports(mode.mode()) {
                    anyhow::bail!(
                        "the metal backend covers create2, create3 and 1nft; profanity needs \
                         a secp256k1 kernel that does not exist for Metal yet, so use \
                         --backend opencl for it"
                    );
                }
                Ok(Box::new(miner_backend::metal::MetalBackend::new()?))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                // Only the arm above reads it, and that arm is compiled out here.
                let _ = mode;
                anyhow::bail!(
                    "metal is only available on macOS builds compiled with --features metal"
                )
            }
        }
        "opencl" => {
            #[cfg(feature = "opencl")]
            {
                if mode.mode() == MineMode::Profanity {
                    return Ok(Box::new(
                        miner_backend::opencl::profanity::ProfanityBackend::new(
                            &tuning.skip_devices,
                        )?,
                    ));
                }
                Ok(Box::new(miner_backend::opencl::salt::SaltBackend::new(
                    &tuning.skip_devices,
                )?))
            }
            #[cfg(not(feature = "opencl"))]
            {
                // As above: a CPU-only build reads neither, and leaving them
                // unmentioned is what made `-D warnings` unreachable for this
                // feature set.
                let _ = (mode, tuning);
                anyhow::bail!("this build was compiled without the opencl feature")
            }
        }
        other => anyhow::bail!("unknown backend {other}"),
    }
}

fn print_devices(
    backend: &dyn Backend,
    common: &CommonArgs,
    mode: &ModeConfig,
    scoring: &ScoringArgs,
) {
    println!("Mode: {} via {}", mode.mode().as_str(), backend.name());
    if backend.name() == "opencl" {
        println!("Kernel: keccak={}", common.kernel);
    }
    println!("Devices:");
    for device in backend.devices() {
        println!(
            "  GPU{}: {}, {} bytes available, {} compute units",
            device.index, device.name, device.global_memory, device.compute_units
        );
    }
    if let ModeConfig::Salt(cfg) = mode {
        println!("Deployer: 0x{}", hex::encode(cfg.deployer));
        if let Some(mint_for) = cfg.mint_for {
            println!("Mint for: 0x{}", hex::encode(mint_for));
        }
    }
    if scoring.is_benchmark() {
        println!("Benchmark: results are throughput only and are not usable addresses.");
    }
    println!();
}
