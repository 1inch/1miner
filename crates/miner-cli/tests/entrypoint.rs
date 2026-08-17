//! The container entrypoint's control flow.
//!
//! `docker/entrypoint.sh` cannot be run as it ships, because it execs an
//! absolute path that exists only inside the image. These tests run the real
//! file with that one path rewritten to a stub, so everything else — the
//! shebang, the passthrough allowlist, `MINER_ARGS`, the `MINER_OUTPUT`
//! pipeline — is exercised as written.
//!
//! Two invariants, both of which `MINER_OUTPUT` broke while it teed through a
//! pipeline: the miner runs exactly once, and the container exits with the
//! miner's own status. A second search costs whatever the box costs per hour,
//! and a status that came from a rerun is not the status of the search that was
//! logged — a hit failing CPU re-derivation is visible only as a non-zero exit.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, ExitStatus};

/// The paths the shipped entrypoint execs, rewritten to stubs in the copy under
/// test.
const MINER_BIN: &str = "/usr/local/bin/1miner";
const BENCH_BIN: &str = "/usr/local/bin/1miner-bench";

/// Stands in for the miner. Records each invocation and writes to both streams
/// so the redirect can be checked on each.
///
/// `STUB_EXIT` applies to the first call only, because that is the shape of the
/// failure the exit status exists to report: a hit that fails CPU re-derivation
/// belongs to one search, and a second search of the same command line can
/// perfectly well exit 0. A stub that failed every time would let the rerun
/// hand back the right status by accident.
const STUB: &str = r#"#!/bin/sh
echo "$*" >> "$STUB_RUNS"
echo "out: $*"
echo "err: $*" >&2
[ -e "$STUB_RUNS.again" ] && exit 0
: > "$STUB_RUNS.again"
exit "${STUB_EXIT:-0}"
"#;

/// Stands in for the shipped `scripts/bench.sh`, so that a test can tell which
/// of the two the entrypoint chose.
const BENCH_STUB: &str = r#"#!/bin/sh
echo "$*" >> "$BENCH_RUNS"
echo "bench: $*"
exit "${BENCH_EXIT:-0}"
"#;

struct Run {
    status: ExitStatus,
    stdout: String,
    /// The argument list the stubbed miner was called with, once per call.
    runs: Vec<String>,
    /// The same for the stubbed bench script.
    bench_runs: Vec<String>,
    /// Contents of `MINER_OUTPUT`, for a run that set it.
    log: Option<String>,
}

/// Run the entrypoint once in a scratch directory of its own. `output` is the
/// `MINER_OUTPUT` path relative to that directory, or `None` to leave the
/// variable unset.
fn run(name: &str, args: &[&str], env: &[(&str, &str)], output: Option<&str>) -> Option<Run> {
    if !usable_shell() {
        return None;
    }

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("entrypoint")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let stub = dir.join("1miner");
    fs::write(&stub, STUB).unwrap();
    fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();

    let bench_stub = dir.join("1miner-bench");
    fs::write(&bench_stub, BENCH_STUB).unwrap();
    fs::set_permissions(&bench_stub, fs::Permissions::from_mode(0o755)).unwrap();

    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docker/entrypoint.sh");
    let script = fs::read_to_string(&source).unwrap();
    assert!(
        script.contains(MINER_BIN),
        "the entrypoint no longer execs {MINER_BIN}"
    );
    assert!(
        script.contains(BENCH_BIN),
        "the entrypoint no longer execs {BENCH_BIN}"
    );
    assert!(
        script.starts_with("#!/bin/bash"),
        "PIPESTATUS carries the miner's status out of the logging pipeline, and sh has no such thing"
    );

    // The bench path first, because the miner's is a prefix of it: the other
    // order rewrites the front of `/usr/local/bin/1miner-bench` and leaves a
    // path that exists nowhere, so every bench assertion below would fail for
    // a reason that has nothing to do with the entrypoint.
    let rewritten = script
        .replace(BENCH_BIN, bench_stub.to_str().unwrap())
        .replace(MINER_BIN, stub.to_str().unwrap());

    // Executed through its own shebang, the way the image runs it.
    let entrypoint = dir.join("entrypoint.sh");
    fs::write(&entrypoint, rewritten).unwrap();
    fs::set_permissions(&entrypoint, fs::Permissions::from_mode(0o755)).unwrap();

    let runs = dir.join("runs");
    let bench_runs = dir.join("bench-runs");
    let log = output.map(|rel| dir.join(rel));

    let mut cmd = Command::new(&entrypoint);
    cmd.args(args)
        .env_remove("MINER_ARGS")
        .env_remove("MINER_OUTPUT")
        // The re-entry guard, in case the suite itself was started from inside
        // a wrapped run.
        .env_remove("MINER_LOG_WRAPPED")
        // clinfo is not on a developer's machine, and the warning it guards is
        // not what these tests are about.
        .env("MINER_SKIP_GPU_CHECK", "1")
        .env("STUB_RUNS", &runs)
        .env("BENCH_RUNS", &bench_runs)
        .envs(env.iter().copied());
    if let Some(path) = &log {
        cmd.env("MINER_OUTPUT", path);
    }
    let out = cmd.output().unwrap();

    let lines = |path: &Path| -> Vec<String> {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    };

    Some(Run {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        runs: lines(&runs),
        bench_runs: lines(&bench_runs),
        log: log.map(|p| fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))),
    })
}

/// A bash that knows PIPESTATUS, which is what carries the miner's status out
/// of the logging pipeline. Somewhere without one cannot run these tests at all,
/// so skip rather than report a failure that does not belong to the script.
///
/// The redirect this replaced needed an openable `/dev/fd` as well, for its
/// process substitution. An ordinary pipeline does not, so one fewer thing about
/// the host has to be true before a logged run works.
fn usable_shell() -> bool {
    match Command::new("bash")
        .args(["-c", "true | true; exit ${PIPESTATUS[0]}"])
        .output()
    {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let why = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "skipping entrypoint test: bash cannot run it: {}",
                why.trim()
            );
            false
        }
        Err(e) => {
            eprintln!("skipping entrypoint test: bash not usable: {e}");
            false
        }
    }
}

/// The regression. `exec 1miner "$@" | tee` replaced only the pipeline's
/// left-hand subshell, so the shell survived to fall through to the exec at the
/// end of the file, and every logged run mined twice.
#[test]
fn miner_output_runs_the_miner_exactly_once() {
    let args = ["create3", "--leading", "0"];
    let Some(run) = run("once", &args, &[], Some("logs/hits.log")) else {
        return;
    };

    assert_eq!(
        run.runs,
        ["create3 --leading 0"],
        "one search, with the given arguments"
    );
    assert!(run.status.success(), "{:?}", run.status);
    assert_eq!(
        run.stdout.matches("out: ").count(),
        1,
        "tee passes the output through once"
    );

    let log = run.log.unwrap();
    assert!(
        log.contains("out: create3 --leading 0"),
        "stdout missing from the log: {log:?}"
    );
    assert!(
        log.contains("err: create3 --leading 0"),
        "stderr missing from the log: {log:?}"
    );
}

/// The pipeline reported tee's status, so `set -e` saw nothing wrong and the
/// status that reached the container came from the rerun rather than from the
/// search that was logged.
#[test]
fn the_miners_exit_status_survives_miner_output() {
    let args = ["create3", "--leading", "0"];
    let env = [("STUB_EXIT", "3")];
    let Some(run) = run("status", &args, &env, Some("hits.log")) else {
        return;
    };

    assert_eq!(
        run.status.code(),
        Some(3),
        "the miner's own status did not get out"
    );
    assert_eq!(
        run.runs.len(),
        1,
        "a failed search was retried: {:?}",
        run.runs
    );
}

/// The path CI's smoke test takes, and the one that was always correct. Here so
/// that a later change to the redirect cannot quietly cost the other branch its
/// exit status.
#[test]
fn the_unlogged_path_keeps_its_status_too() {
    let args = ["create3", "--leading", "0"];
    let env = [("STUB_EXIT", "2")];
    let Some(run) = run("plain", &args, &env, None) else {
        return;
    };

    assert_eq!(run.status.code(), Some(2));
    assert_eq!(run.runs.len(), 1, "{:?}", run.runs);
}

/// `sh -c '1miner self-test && 1miner ...'`, which docs/vastai.md recommends,
/// leaves through the passthrough. The redirect used to sit below it, so that
/// form logged nothing at all and said so nowhere.
#[test]
fn miner_output_covers_the_passthrough_form() {
    let args = ["sh", "-c", "echo from-passthrough; exit 4"];
    let Some(run) = run("passthrough", &args, &[], Some("hits.log")) else {
        return;
    };

    assert_eq!(
        run.status.code(),
        Some(4),
        "the passthrough lost its exit status"
    );
    assert!(
        run.runs.is_empty(),
        "the miner ran when it should not have: {:?}",
        run.runs
    );
    let log = run.log.unwrap();
    assert!(
        log.contains("from-passthrough"),
        "the passthrough was not logged: {log:?}"
    );
}

/// `MINER_ARGS` is the only way to configure a run on a hosting panel that
/// offers an image and an environment but no command line.
#[test]
fn miner_args_reaches_the_miner_once() {
    let env = [("MINER_ARGS", "create2 --leading 0")];
    let Some(run) = run("miner_args", &[], &env, Some("hits.log")) else {
        return;
    };

    assert_eq!(run.runs, ["create2 --leading 0"]);
    assert!(run.bench_runs.is_empty(), "{:?}", run.bench_runs);
    let log = run.log.unwrap();
    assert!(log.contains("out: create2 --leading 0"), "{log:?}");
}

/// `bench` is the shipped benchmark script rather than a subcommand of the
/// binary, so it needs both an allowlist entry and an arm of its own. With
/// neither it reached the passthrough and the container exec'd a command called
/// `bench`, which exists nowhere in the image.
#[test]
fn bench_reaches_the_script_and_not_the_miner() {
    let Some(run) = run("bench", &["bench", "--balanced"], &[], Some("bench.log")) else {
        return;
    };

    assert_eq!(run.bench_runs, ["--balanced"], "the subcommand is consumed");
    assert!(
        run.runs.is_empty(),
        "the miner was run directly: {:?}",
        run.runs
    );
    assert!(run.status.success(), "{:?}", run.status);
    let log = run.log.unwrap();
    assert!(log.contains("bench: --balanced"), "{log:?}");
}

/// Benchmarking is most worth automating on exactly the panels that offer no
/// command line, so the dispatch sits below the `MINER_ARGS` fallback rather
/// than beside the passthrough.
#[test]
fn bench_reaches_the_script_from_miner_args_too() {
    let env = [("MINER_ARGS", "bench --fast -M create3")];
    let Some(run) = run("bench_args", &[], &env, None) else {
        return;
    };

    assert_eq!(run.bench_runs, ["--fast -M create3"]);
    assert!(run.runs.is_empty(), "{:?}", run.runs);
}

/// A benchmark that ran on a device failing its self-test is worth nothing, and
/// the script exits non-zero to say so. That has to survive the entrypoint for
/// the same reason a search's status does.
#[test]
fn the_bench_scripts_exit_status_survives() {
    let env = [("BENCH_EXIT", "1")];
    let Some(run) = run("bench_status", &["bench"], &env, Some("bench.log")) else {
        return;
    };

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(run.bench_runs.len(), 1, "{:?}", run.bench_runs);
}
