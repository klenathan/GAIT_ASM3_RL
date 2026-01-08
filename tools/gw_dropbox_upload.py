#!/usr/bin/env python3
"""Upload a single run directory to Dropbox (no rclone).

This tool uses the official Dropbox Python SDK.

Auth model: long-lived access token (simplest).

Environment variables
---------------------
DROPBOX_ACCESS_TOKEN
    Long-lived Dropbox access token.

Examples
--------
# Upload a run directory as a .zip into /asm3/runs/gridworld/
uv run python tools/gw_dropbox_upload.py \
  --run runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld" \
  --token-env DROPBOX_ACCESS_TOKEN

# Dry-run (shows what would be uploaded)
uv run python tools/gw_dropbox_upload.py --run runs/gridworld/<run_name> --dest "/asm3/runs/gridworld" --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UploadSpec:
    run_dir: Path
    dropbox_dest_dir: str
    overwrite: bool
    dry_run: bool


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gw_dropbox_upload",
        description="Upload one run directory to Dropbox as a zip (no rclone).",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Local run directory to upload (e.g. runs/gridworld/<run_name>)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help='Dropbox destination folder, e.g. "/asm3/runs/gridworld" (must start with /)',
    )

    token_group = parser.add_mutually_exclusive_group(required=False)
    token_group.add_argument(
        "--token",
        default=None,
        help="Dropbox access token (not recommended to pass on CLI)",
    )
    token_group.add_argument(
        "--token-file",
        default=None,
        help="Path to file containing Dropbox access token",
    )
    token_group.add_argument(
        "--token-env",
        default="DROPBOX_ACCESS_TOKEN",
        help="Env var name containing Dropbox token (default: DROPBOX_ACCESS_TOKEN)",
    )

    parser.add_argument(
        "--zip-name",
        default=None,
        help="Optional zip file name (default: <run_name>.zip)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file in Dropbox if present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without uploading",
    )

    return parser.parse_args(argv)


def _load_token(args: argparse.Namespace) -> str:
    if args.token:
        return str(args.token).strip()

    if args.token_file:
        token_path = Path(args.token_file).expanduser().resolve()
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"Token file is empty: {token_path}")
        return token

    env_name = str(args.token_env or "DROPBOX_ACCESS_TOKEN")
    token = os.getenv(env_name, "").strip()
    if not token:
        raise ValueError(
            f"Missing Dropbox token. Set ${env_name} or pass --token-file/--token."
        )
    return token


def _assert_dropbox_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"Dropbox paths must start with '/': {path}")
    # Normalize trailing slash
    return path.rstrip("/")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip_run_dir(run_dir: Path, zip_path: Path) -> None:
    # Keep run folder name at top-level inside the zip.
    base = run_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_dir():
                continue
            rel = file_path.relative_to(base)
            zf.write(file_path, arcname=str(rel))


def _dropbox_upload(
    zip_path: Path, token: str, dest_path: str, overwrite: bool
) -> None:
    import dropbox  # imported lazily so --help works even before deps

    dbx = dropbox.Dropbox(token)

    mode = (
        dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
    )

    with zip_path.open("rb") as f:
        data = f.read()

    # Single-call upload is fine for typical run sizes. If you expect multi-GB
    # zips, we can switch to upload sessions.
    dbx.files_upload(data, dest_path, mode=mode)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"--run must be a directory: {run_dir}")

    dest_dir = _assert_dropbox_path(str(args.dest))

    zip_name = args.zip_name or f"{run_dir.name}.zip"
    if not zip_name.endswith(".zip"):
        zip_name += ".zip"

    dropbox_path = f"{dest_dir}/{zip_name}"

    spec = UploadSpec(
        run_dir=run_dir,
        dropbox_dest_dir=dest_dir,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    print("Local run:", str(spec.run_dir))
    print("Dropbox file:", dropbox_path)

    with tempfile.TemporaryDirectory(prefix="gw_dropbox_") as tmp:
        zip_path = Path(tmp) / zip_name
        _zip_run_dir(spec.run_dir, zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        sha = _sha256_file(zip_path)
        print(f"Prepared zip: {zip_path.name} ({size_mb:.2f} MB)")
        print(f"SHA256: {sha}")

        if spec.dry_run:
            print("Dry-run: not uploading.")
            return 0

        token = _load_token(args)
        _dropbox_upload(
            zip_path, token=token, dest_path=dropbox_path, overwrite=spec.overwrite
        )
        print("Upload complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
