//! The container entrypoint's control flow.
//!
//! `docker/entrypoint.sh` cannot be run as it ships, because it execs an
//! absolute path that exists only inside the image. These tests run the real
//! file with that one path rewritten to a stub, so everything else — the
//! shebang, the passthrough allowlist, `MINER_ARGS`, the `MINER_OUTPUT`
//! redirect — is exercised as written.
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

/// The path the shipped entrypoint execs, rewritten to the stub in the copy
/// under test.
const MINER_BIN: &str = "/usr/local/bin/1miner";

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

struct Run {
    status: ExitStatus,
    stdout: String,
    /// The argument list the stubbed miner was called with, once per call.
    runs: Vec<String>,
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

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("entrypoint").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let stub = dir.join("1miner");
    fs::write(&stub, STUB).unwrap();
    fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();

    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docker/entrypoint.sh");
    let script = fs::read_to_string(&source).unwrap();
    assert!(script.contains(MINER_BIN), "the entrypoint no longer execs {MINER_BIN}");
    assert!(
        script.starts_with("#!/bin/bash"),
        "the MINER_OUTPUT redirect is a bash process substitution, which sh cannot run"
    );

    // Executed through its own shebang, the way the image runs it.
    let entrypoint = dir.join("entrypoint.sh");
    fs::write(&entrypoint, script.replace(MINER_BIN, stub.to_str().unwrap())).unwrap();
    fs::set_permissions(&entrypoint, fs::Permissions::from_mode(0o755)).unwrap();

    let runs = dir.join("runs");
    let log = output.map(|rel| dir.join(rel));

    let mut cmd = Command::new(&entrypoint);
    cmd.args(args)
        .env_remove("MINER_ARGS")
        .env_remove("MINER_OUTPUT")
        // clinfo is not on a developer's machine, and the warning it guards is
        // not what these tests are about.
        .env("MINER_SKIP_GPU_CHECK", "1")
        .env("STUB_RUNS", &runs)
        .envs(env.iter().copied());
    if let Some(path) = &log {
        cmd.env("MINER_OUTPUT", path);
    }
    let out = cmd.output().unwrap();

    Some(Run {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        runs: fs::read_to_string(&runs).unwrap_or_default().lines().map(str::to_owned).collect(),
        log: log.map(|p| {
            fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        }),
    })
}

/// bash and an openable `/dev/fd`, which is what `exec 1> >(tee ...)` needs. A
/// sandbox that forbids `/dev/fd` cannot run these tests at all, so skip rather
/// than report a failure that does not belong to the script.
fn usable_shell() -> bool {
    match Command::new("bash").args(["-c", "exec 1> >(cat >/dev/null)"]).output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let why = String::from_utf8_lossy(&out.stderr);
            eprintln!("skipping entrypoint test: no process substitution here: {}", why.trim());
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
    let Some(run) = run("once", &args, &[], Some("logs/hits.log")) else { return };

    assert_eq!(run.runs, ["create3 --leading 0"], "one search, with the given arguments");
    assert!(run.status.success(), "{:?}", run.status);
    assert_eq!(run.stdout.matches("out: ").count(), 1, "tee passes the output through once");

    let log = run.log.unwrap();
    assert!(log.contains("out: create3 --leading 0"), "stdout missing from the log: {log:?}");
    assert!(log.contains("err: create3 --leading 0"), "stderr missing from the log: {log:?}");
}

/// The pipeline reported tee's status, so `set -e` saw nothing wrong and the
/// status that reached the container came from the rerun rather than from the
/// search that was logged.
#[test]
fn the_miners_exit_status_survives_miner_output() {
    let args = ["create3", "--leading", "0"];
    let env = [("STUB_EXIT", "3")];
    let Some(run) = run("status", &args, &env, Some("hits.log")) else { return };

    assert_eq!(run.status.code(), Some(3), "the miner's own status did not get out");
    assert_eq!(run.runs.len(), 1, "a failed search was retried: {:?}", run.runs);
}

/// The path CI's smoke test takes, and the one that was always correct. Here so
/// that a later change to the redirect cannot quietly cost the other branch its
/// exit status.
#[test]
fn the_unlogged_path_keeps_its_status_too() {
    let args = ["create3", "--leading", "0"];
    let env = [("STUB_EXIT", "2")];
    let Some(run) = run("plain", &args, &env, None) else { return };

    assert_eq!(run.status.code(), Some(2));
    assert_eq!(run.runs.len(), 1, "{:?}", run.runs);
}

/// `sh -c '1miner self-test && 1miner ...'`, which docs/vastai.md recommends,
/// leaves through the passthrough. The redirect used to sit below it, so that
/// form logged nothing at all and said so nowhere.
#[test]
fn miner_output_covers_the_passthrough_form() {
    let args = ["sh", "-c", "echo from-passthrough; exit 4"];
    let Some(run) = run("passthrough", &args, &[], Some("hits.log")) else { return };

    assert_eq!(run.status.code(), Some(4), "the passthrough lost its exit status");
    assert!(run.runs.is_empty(), "the miner ran when it should not have: {:?}", run.runs);
    let log = run.log.unwrap();
    assert!(log.contains("from-passthrough"), "the passthrough was not logged: {log:?}");
}

/// `MINER_ARGS` is the only way to configure a run on a hosting panel that
/// offers an image and an environment but no command line.
#[test]
fn miner_args_reaches_the_miner_once() {
    let env = [("MINER_ARGS", "create2 --leading 0")];
    let Some(run) = run("miner_args", &[], &env, Some("hits.log")) else { return };

    assert_eq!(run.runs, ["create2 --leading 0"]);
    let log = run.log.unwrap();
    assert!(log.contains("out: create2 --leading 0"), "{log:?}");
}
