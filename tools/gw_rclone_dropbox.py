#!/usr/bin/env python3
"""Upload runs to Dropbox using rclone.

This is a small convenience wrapper around `rclone copy/sync` specifically for
Dropbox, for a common "SSH-only" workflow:

- You SSH into the school machine
- Your host machine does the browser auth via `rclone authorize "dropbox"`
- You paste the resulting token back into the school machine's `rclone config`

This script itself needs only:
- python3
- rclone

Examples
--------
# Upload a single run directory
python3 tools/gw_rclone_dropbox.py \
  --remote dropbox \
  --dest "asm3" \
  --run runs/gridworld/<run_name>

# Upload all gridworld runs
python3 tools/gw_rclone_dropbox.py --remote dropbox --dest "asm3" --what gridworld

# Dry run
python3 tools/gw_rclone_dropbox.py --remote dropbox --dest "asm3" --what gridworld --dry-run
"""

from __future__ import annotations

import argparse
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


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_runs_dir() -> Path:
    return _repo_root_from_this_file() / "runs"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gw_rclone_dropbox",
        description="Upload training runs to Dropbox using rclone.",
    )

    parser.add_argument(
        "--remote",
        default="dropbox",
        help="rclone remote name for Dropbox (default: dropbox)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination folder inside Dropbox remote (e.g. 'asm3')",
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

        try:
            rel = source.relative_to(runs_dir)
        except ValueError:
            rel = Path("manual") / source.name

        target = RcloneTarget(
            remote=args.remote,
            dest=(Path(args.dest) / "runs" / rel.as_posix()).as_posix(),
        )

    else:
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

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
