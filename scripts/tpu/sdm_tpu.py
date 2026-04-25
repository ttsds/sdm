#!/usr/bin/env python3
"""sdm-tpu: small CLI to request and manage TRC TPU VMs for the sdm project.

Subcommands:
  allocations          List the TRC zones/accelerators we know about.
  request   [pool]     Start one or more background poll loops that retry
                       `gcloud compute tpus tpu-vm create` until capacity opens.
  status               Show running poll loops and their latest log line.
  list                 List existing TPU VMs across all known zones.
  ssh       <name>     SSH into a created VM (auto-detect zone).
  delete    <name>     Delete a VM (auto-detect zone).
  stop      [name]     Kill the local poll loop(s). With no arg: kill all.
  tail      [name]     Tail a poll loop's log; with no arg, multiplex all.

Pools (presets matching the TRC welcome email for project ml-edinburgh):
  v4-od            on-demand v4-8 in us-central2-b (32 chips OD)
  v4-spot          spot v4-8 in us-central2-b (32 chips spot)
  v5e-usc1         spot v5litepod-8 in us-central1-a (64 chips)
  v5e-euw4         spot v5litepod-8 in us-central1-a (64 chips)  [in europe-west4-b]
  v6e-use1         spot v6e-8 in us-east1-d (64 chips)
  v6e-euw4         spot v6e-8 in europe-west4-a (64 chips)
  all              every pool in parallel
  spot             every spot pool in parallel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT = os.environ.get("SDM_GCP_PROJECT", "ml-edinburgh")
LOG_DIR = Path(os.environ.get("SDM_TPU_LOG_DIR", "/tmp/sdm_tpu"))
PID_DIR = LOG_DIR / "pids"
GCLOUD = os.environ.get("GCLOUD", "gcloud")


@dataclass(frozen=True)
class Pool:
    name: str         # short id, used for the VM name suffix and log file
    accel: str        # gcloud --accelerator-type
    zone: str         # gcloud --zone
    spot: bool        # whether to pass --spot

    @property
    def runtime(self) -> str:
        if self.accel.startswith("v4-"):
            return "tpu-ubuntu2204-base"
        if self.accel.startswith("v5e-") or self.accel.startswith("v5litepod-"):
            return "v2-alpha-tpuv5-lite"
        if self.accel.startswith("v6e-"):
            return "v2-alpha-tpuv6e"
        raise ValueError(f"unknown accelerator family: {self.accel}")


# Keep this in sync with the TRC welcome email.
POOLS: dict[str, Pool] = {
    "v4-od":     Pool("v4-od",     "v4-8",        "us-central2-b",  spot=False),
    "v4-spot":   Pool("v4-spot",   "v4-8",        "us-central2-b",  spot=True),
    "v5e-usc1":  Pool("v5e-usc1",  "v5litepod-8", "us-central1-a",  spot=True),
    "v5e-euw4":  Pool("v5e-euw4",  "v5litepod-8", "europe-west4-b", spot=True),
    "v6e-use1":  Pool("v6e-use1",  "v6e-8",       "us-east1-d",     spot=True),
    "v6e-euw4":  Pool("v6e-euw4",  "v6e-8",       "europe-west4-a", spot=True),
}

POOL_GROUPS: dict[str, list[str]] = {
    "all":  list(POOLS),
    "spot": [k for k, p in POOLS.items() if p.spot],
}

RETRYABLE = re.compile(
    r"no more capacity|Insufficient capacity|RESOURCE_EXHAUSTED|UNAVAILABLE|"
    r"resourceExhausted|Stockout|currently unavailable|tenant project creation|"
    r'"code": 8|"code": 10|HttpError|503|504|deadline exceeded|Internal error',
    re.IGNORECASE,
)


def log_path(pool: Pool) -> Path:
    return LOG_DIR / f"poll_{pool.name}.log"


def pid_path(pool: Pool) -> Path:
    return PID_DIR / f"poll_{pool.name}.pid"


def vm_name(prefix: str, pool: Pool) -> str:
    return f"{prefix}-{pool.name}"


# ----------------------------------------------------------------------------
# Poll-loop worker. When invoked with `--worker <pool> <vm-name>`, this same
# script becomes the actual retry loop (so we don't need a separate shell
# script). `request` daemonises copies of itself in worker mode.
# ----------------------------------------------------------------------------
def run_worker(pool_name: str, name: str) -> int:
    pool = POOLS[pool_name]
    log = log_path(pool)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)

    def write(line: str) -> None:
        with open(log, "a") as f:
            f.write(line + "\n")
            f.flush()

    write(
        f"[poll] start pool={pool.name} accel={pool.accel} zone={pool.zone} "
        f"spot={pool.spot} runtime={pool.runtime} vm_name={name}"
    )
    cmd = [
        GCLOUD, "compute", "tpus", "tpu-vm", "create", name,
        f"--project={PROJECT}",
        f"--zone={pool.zone}",
        f"--accelerator-type={pool.accel}",
        f"--version={pool.runtime}",
    ]
    if pool.spot:
        cmd.append("--spot")

    attempt = 0
    while True:
        attempt += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write(f"[poll] {ts} attempt {attempt}: {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode == 0:
            write(f"[poll] up after {attempt} attempts.")
            write(proc.stdout)
            write(
                f"  {GCLOUD} compute tpus tpu-vm ssh {name} "
                f"--project={PROJECT} --zone={pool.zone} --worker=all"
            )
            return 0
        # Combined stderr+stdout; keep last few lines for the log.
        err = (proc.stderr or "") + (proc.stdout or "")
        tail = "\n".join(err.strip().splitlines()[-6:])
        write(tail)
        if RETRYABLE.search(err):
            write(f"[poll] retryable error; retrying (attempt {attempt + 1})")
        else:
            write(f"[poll] unrecognised error; retrying anyway (Ctrl-C / sdm-tpu stop to abort)")
        # gcloud already blocks long enough; no extra sleep needed.


# ----------------------------------------------------------------------------
# CLI commands.
# ----------------------------------------------------------------------------
def cmd_allocations(_args: argparse.Namespace) -> int:
    print(f"{'POOL':<10} {'ACCEL':<14} {'ZONE':<16} SPOT")
    for pool in POOLS.values():
        print(f"{pool.name:<10} {pool.accel:<14} {pool.zone:<16} {pool.spot}")
    return 0


def _resolve_pools(args: argparse.Namespace) -> list[Pool]:
    pools: list[Pool] = []
    for token in args.pools:
        if token in POOL_GROUPS:
            for p in POOL_GROUPS[token]:
                pools.append(POOLS[p])
        elif token in POOLS:
            pools.append(POOLS[token])
        else:
            raise SystemExit(
                f"unknown pool '{token}'. Available: {', '.join(POOLS)} "
                f"or groups: {', '.join(POOL_GROUPS)}"
            )
    # de-dup, preserving order
    seen: set[str] = set()
    unique: list[Pool] = []
    for p in pools:
        if p.name in seen:
            continue
        seen.add(p.name)
        unique.append(p)
    return unique


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_request(args: argparse.Namespace) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    pools = _resolve_pools(args)
    if not pools:
        print("no pools selected", file=sys.stderr)
        return 2
    script = os.path.abspath(__file__)
    started: list[tuple[Pool, int, str]] = []
    for pool in pools:
        pidfile = pid_path(pool)
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                if _is_alive(pid):
                    print(f"[skip] pool {pool.name}: already running (pid {pid})")
                    continue
            except ValueError:
                pass
        name = vm_name(args.prefix, pool)
        log = log_path(pool)
        # truncate the log on (re)start so each request starts fresh
        log.write_text("")
        proc = subprocess.Popen(
            [sys.executable, script, "--worker", pool.name, name],
            stdout=open(log, "a"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pidfile.write_text(str(proc.pid))
        started.append((pool, proc.pid, name))
        print(f"[ok] pool {pool.name:<9} pid {proc.pid:<7} vm {name} log {log}")
    if not started:
        print("nothing to do (all pools already running)")
    return 0


def _read_pid(pool: Pool) -> int | None:
    p = pid_path(pool)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def _last_log_lines(pool: Pool, n: int) -> list[str]:
    p = log_path(pool)
    if not p.exists():
        return []
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return []
    return text.splitlines()[-n:]


def cmd_status(args: argparse.Namespace) -> int:
    print(f"{'POOL':<10} {'PID':<8} {'STATE':<8} {'ATTEMPT':<8} LAST")
    for pool in POOLS.values():
        pid = _read_pid(pool)
        if pid is None:
            state, pid_str = "idle", "-"
        elif _is_alive(pid):
            state, pid_str = "alive", str(pid)
        else:
            state, pid_str = "dead", str(pid)
        attempt = "-"
        last = ""
        for line in reversed(_last_log_lines(pool, 200)):
            m = re.search(r"attempt (\d+)", line)
            if m and attempt == "-":
                attempt = m.group(1)
            if not last and line.strip():
                last = line.strip()
            if attempt != "-" and last:
                break
        print(f"{pool.name:<10} {pid_str:<8} {state:<8} {attempt:<8} {last[:120]}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    targets = _resolve_pools(args) if args.pools else list(POOLS.values())
    killed = 0
    for pool in targets:
        pid = _read_pid(pool)
        if pid is None or not _is_alive(pid):
            continue
        try:
            # Kill the process group so gcloud children die too.
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed += 1
            print(f"[stop] pool {pool.name}: killed pid {pid}")
        except (ProcessLookupError, PermissionError) as exc:
            print(f"[stop] pool {pool.name}: {exc}", file=sys.stderr)
        finally:
            pid_path(pool).unlink(missing_ok=True)
    print(f"stopped {killed} loop(s)")
    return 0


def _list_zone(zone: str) -> list[dict]:
    proc = subprocess.run(
        [GCLOUD, "compute", "tpus", "tpu-vm", "list",
         f"--project={PROJECT}", f"--zone={zone}", "--format=json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


def cmd_list(args: argparse.Namespace) -> int:
    zones = sorted({p.zone for p in POOLS.values()})
    found = []
    for zone in zones:
        for vm in _list_zone(zone):
            found.append((zone, vm))
    if not found:
        print("(no TPU VMs found in any TRC zone for project " + PROJECT + ")")
        return 0
    print(f"{'NAME':<28} {'ZONE':<16} {'ACCEL':<14} {'STATE':<10} {'CREATED'}")
    for zone, vm in found:
        name = vm.get("name", "?").split("/")[-1]
        accel = vm.get("acceleratorType", "?").split("/")[-1]
        state = vm.get("state", "?")
        created = vm.get("createTime", "?")
        print(f"{name:<28} {zone:<16} {accel:<14} {state:<10} {created}")
    return 0


def _find_zone_for(name: str) -> str | None:
    for pool in POOLS.values():
        for vm in _list_zone(pool.zone):
            if vm.get("name", "").split("/")[-1] == name:
                return pool.zone
    return None


def cmd_ssh(args: argparse.Namespace) -> int:
    zone = args.zone or _find_zone_for(args.name)
    if not zone:
        print(f"VM {args.name} not found in any known zone", file=sys.stderr)
        return 1
    cmd = [GCLOUD, "compute", "tpus", "tpu-vm", "ssh", args.name,
           f"--project={PROJECT}", f"--zone={zone}", "--worker=all"]
    if args.command:
        cmd += ["--command", args.command]
    return subprocess.call(cmd)


def cmd_delete(args: argparse.Namespace) -> int:
    zone = args.zone or _find_zone_for(args.name)
    if not zone:
        print(f"VM {args.name} not found in any known zone", file=sys.stderr)
        return 1
    cmd = [GCLOUD, "compute", "tpus", "tpu-vm", "delete", args.name,
           f"--project={PROJECT}", f"--zone={zone}", "--quiet"]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_tail(args: argparse.Namespace) -> int:
    pools = _resolve_pools(args) if args.pools else list(POOLS.values())
    files = [str(log_path(p)) for p in pools if log_path(p).exists()]
    if not files:
        print("no log files yet")
        return 0
    cmd = ["tail", "-n", str(args.n), "-F", *files] if args.follow else ["tail", "-n", str(args.n), *files]
    return subprocess.call(cmd)


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sdm-tpu", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", nargs=2, metavar=("POOL", "NAME"),
                   help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("allocations", help="list known TRC zones/accelerators")

    pr = sub.add_parser("request", help="start poll loops in the background")
    pr.add_argument("pools", nargs="+",
                    help="pool names or groups (e.g. v4-od v6e-euw4 / all / spot)")
    pr.add_argument("--prefix", default="sdm-test", help="VM name prefix")

    st = sub.add_parser("status", help="show running poll loops")
    _ = st

    sp = sub.add_parser("stop", help="kill poll loops")
    sp.add_argument("pools", nargs="*", help="pool names or groups; default = all")

    sub.add_parser("list", help="list TPU VMs in known zones")

    ssh = sub.add_parser("ssh", help="ssh into a TPU VM")
    ssh.add_argument("name")
    ssh.add_argument("--zone", default=None)
    ssh.add_argument("--command", default=None)

    dl = sub.add_parser("delete", help="delete a TPU VM")
    dl.add_argument("name")
    dl.add_argument("--zone", default=None)

    tl = sub.add_parser("tail", help="tail poll-loop logs")
    tl.add_argument("pools", nargs="*", help="pool names or groups; default = all")
    tl.add_argument("-n", type=int, default=20)
    tl.add_argument("-f", "--follow", action="store_true")

    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.worker:
        return run_worker(args.worker[0], args.worker[1])

    handlers = {
        "allocations": cmd_allocations,
        "request":     cmd_request,
        "status":      cmd_status,
        "stop":        cmd_stop,
        "list":        cmd_list,
        "ssh":         cmd_ssh,
        "delete":      cmd_delete,
        "tail":        cmd_tail,
    }
    if args.cmd is None:
        parser.print_help()
        return 0
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
