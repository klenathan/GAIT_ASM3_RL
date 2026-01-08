#!/usr/bin/env python3
"""Upload GridWorld (and other) run artifacts via rclone.

Designed for school Linux machines without root, where `rclone` is already
available. This script intentionally does not depend on any Python packages.

Examples
--------
# Upload all gridworld runs (copy new/changed only)
python tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld

# Upload a single run folder
python tools/gw_upload.py --remote proton --dest "school/asm3" --run runs/gridworld/<run_name>

# Preview what would be uploaded
python tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RcloneTarget:
    remote: str
    dest: str

    @property
    def remote_path(self) -> str:
        if not self.dest:
            return f"{self.remote}:"
        return f"{self.remote}:{self.dest.lstrip('/')}"


def _ensure_rclone_exists() -> None:
    if shutil.which("rclone") is None:
        raise FileNotFoundError(
            "rclone not found in PATH. Install it or ask admin to provide it."
        )


def _run(cmd: list[str]) -> int:
    # Stream output directly so it works over SSH.
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _repo_root_from_this_file() -> Path:
    # tools/gw_upload.py -> repo root
    return Path(__file__).resolve().parents[1]


def _default_runs_dir() -> Path:
    return _repo_root_from_this_file() / "runs"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gw_upload",
        description="Upload training runs to Proton Drive using rclone.",
    )

    parser.add_argument(
        "--remote",
        required=True,
        help="rclone remote name (e.g. 'proton')",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination folder inside the remote (e.g. 'school/asm3')",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--what",
        choices=["gridworld", "all"],
        help="Upload all runs for a category (from ./runs)",
    )
    src_group.add_argument(
        "--run",
        help="Upload a specific run directory (e.g. runs/gridworld/<run_name>)",
    )

    parser.add_argument(
        "--runs_dir",
        default=str(_default_runs_dir()),
        help="Local runs directory (default: ./runs)",
    )

    parser.add_argument(
        "--mode",
        choices=["copy", "sync"],
        default="copy",
        help="Upload mode: copy leaves remote extras; sync deletes extras.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without uploading",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show detailed progress (rclone -P)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra rclone --exclude patterns (repeatable)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra args passed to rclone (repeatable; e.g. --extra='--bwlimit=8M')",
    )

    return parser.parse_args(argv)


def _build_rclone_cmd(
    *,
    mode: str,
    source: Path,
    target: RcloneTarget,
    dry_run: bool,
    progress: bool,
    excludes: list[str],
    extra: list[str],
) -> list[str]:
    cmd: list[str] = ["rclone", mode, str(source), target.remote_path]

    # Reasonable defaults for logs/artifacts.
    cmd += [
        "--create-empty-src-dirs",
        "--copy-links",
        "--metadata",
        "--modify-window",
        "2s",
        "--exclude",
        "**/__pycache__/**",
        "--exclude",
        "**/.DS_Store",
    ]

    for pattern in excludes:
        cmd += ["--exclude", pattern]

    if dry_run:
        cmd.append("--dry-run")

    if progress:
        cmd.append("-P")
    else:
        cmd += ["--stats", "10s"]

    for item in extra:
        # allow passing a single token or multi-token string
        cmd += item.split()

    return cmd


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    _ensure_rclone_exists()

    runs_dir = Path(args.runs_dir).expanduser().resolve()

    if args.run:
        source = Path(args.run).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Run directory not found: {source}")

        # Remote destination mirrors `runs/...` structure for consistency.
        try:
            rel = source.relative_to(runs_dir)
        except ValueError:
            # If user passes an absolute path outside runs_dir, still upload it,
            # but put it under a safety folder.
            rel = Path("manual") / source.name

        dest = (Path(args.dest) / "runs" / rel.as_posix()).as_posix()
        target = RcloneTarget(remote=args.remote, dest=dest)

    else:
        # Upload by category from runs_dir
        if not runs_dir.exists():
            raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

        if args.what == "gridworld":
            source = runs_dir / "gridworld"
            if not source.exists():
                raise FileNotFoundError(f"No gridworld runs found at: {source}")
            target = RcloneTarget(
                remote=args.remote, dest=f"{args.dest}/runs/gridworld"
            )
        elif args.what == "all":
            source = runs_dir
            target = RcloneTarget(remote=args.remote, dest=f"{args.dest}/runs")
        else:
            raise ValueError(f"Unsupported --what: {args.what}")

    mode = "copy" if args.mode == "copy" else "sync"
    cmd = _build_rclone_cmd(
        mode=mode,
        source=source,
        target=target,
        dry_run=bool(args.dry_run),
        progress=bool(args.progress),
        excludes=list(args.exclude),
        extra=list(args.extra),
    )

    print("Local source:", str(source))
    print("Remote dest:", target.remote_path)
    print("Command:", " ".join(cmd))

    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
