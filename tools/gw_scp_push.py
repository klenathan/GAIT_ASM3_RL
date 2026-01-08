#!/usr/bin/env python3
"""Push a single run directory from a remote machine via scp.

This is intended for school machines where you can SSH out to your own host
machine. It requires only:
- python3
- ssh/scp

Examples
--------
# Push a single run to your host machine
python3 tools/gw_scp_push.py \
  --run runs/gridworld/<run_name> \
  --host user@my-laptop \
  --dest "~/asm3_uploads/GAIT_ASM3_RL"

# Same, with SSH key + compression
python3 tools/gw_scp_push.py \
  --run runs/gridworld/<run_name> \
  --host user@my-laptop \
  --dest "~/asm3_uploads/GAIT_ASM3_RL" \
  --identity ~/.ssh/id_ed25519 \
  --compress

# Show the scp command without executing it
python3 tools/gw_scp_push.py --run runs/gridworld/<run_name> --host user@my-laptop --dest "~/asm3" --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_tool_exists(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"{name} not found in PATH")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gw_scp_push",
        description="Push one training run directory to your host machine using scp.",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Local run directory to upload (e.g. runs/gridworld/<run_name>)",
    )
    parser.add_argument(
        "--host",
        required=True,
        help="SSH destination in scp format (e.g. user@host or host)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help=(
            "Destination base directory on host machine (e.g. ~/asm3_uploads). "
            "The run folder name will be created underneath."
        ),
    )

    parser.add_argument("--port", type=int, default=None, help="SSH port")
    parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Path to SSH private key (passed to scp -i)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable compression (scp -C)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scp command without executing",
    )

    return parser.parse_args(argv)


def _scp_command(
    *,
    source: Path,
    host: str,
    dest_base: str,
    port: int | None,
    identity: str | None,
    compress: bool,
) -> list[str]:
    # We copy the *directory* source into dest_base, i.e.
    #   scp -r runs/gridworld/foo user@host:~/asm3_uploads/GAIT_ASM3_RL/
    # which results in:
    #   ~/asm3_uploads/GAIT_ASM3_RL/foo/
    cmd: list[str] = ["scp", "-r"]

    if compress:
        cmd.append("-C")

    if port is not None:
        cmd += ["-P", str(port)]

    if identity is not None:
        cmd += ["-i", identity]

    cmd += [str(source), f"{host}:{dest_base.rstrip('/')}/"]
    return cmd


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    _ensure_tool_exists("scp")

    source = Path(args.run).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Run directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"--run must be a directory: {source}")

    # scp cannot expand ~ on the remote side reliably if quoted, but it *does*
    # expand when it reaches the remote shell. So we leave dest as provided.
    dest_base = args.dest

    cmd = _scp_command(
        source=source,
        host=args.host,
        dest_base=dest_base,
        port=args.port,
        identity=args.identity,
        compress=bool(args.compress),
    )

    print("Local source:", str(source))
    print("Remote dest:", f"{args.host}:{dest_base.rstrip('/')}/")
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
