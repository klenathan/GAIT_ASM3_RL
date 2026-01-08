#!/usr/bin/env python3
"""Upload files or folders to Dropbox (no rclone).

This tool uses the official Dropbox Python SDK.

Auth model: long-lived access token (simplest).

Environment variables
---------------------
DROPBOX_ACCESS_TOKEN
    Long-lived Dropbox access token.

What it can upload
------------------
- A single file: upload to a Dropbox path
- A directory: either
  - zip it and upload as a single `.zip` (default), or
  - upload the directory tree (one file at a time)

Backwards compatibility
-----------------------
The old `--run <dir>` behaviour is still supported as an alias for
`--src <dir> --zip`.

Examples
--------
# Upload a run directory as a .zip into /asm3/runs/gridworld/
uv run python tools/gw_dropbox_upload.py \
  --run runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld"

# Upload an arbitrary file
uv run python tools/gw_dropbox_upload.py \
  --src runs/gridworld/<run_name>/final/model.pkl \
  --dest "/asm3/models"

# Upload a directory without zipping (mirrors directory structure)
uv run python tools/gw_dropbox_upload.py \
  --src runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld/<run_name>" \
  --no-zip

# Dry-run
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    dropbox = object()  # type: ignore


@dataclass(frozen=True)
class UploadSpec:
    source: Path
    dropbox_dest: str
    zip_dir: bool
    overwrite: bool
    dry_run: bool


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gw_dropbox_upload",
        description="Upload a file or directory to Dropbox (no rclone).",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--src",
        help="Local file or directory to upload",
    )
    # Back-compat alias for old workflow.
    src_group.add_argument(
        "--run",
        help="Alias for --src (kept for runs/gridworld/<run_name> usage)",
    )

    parser.add_argument(
        "--dest",
        required=True,
        help=(
            "Dropbox destination path. If uploading a file: destination folder. "
            "If uploading a directory without zipping: destination directory (full path). "
            'Examples: "/asm3/models" or "/asm3/runs/gridworld/<run_name>"'
        ),
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

    zip_group = parser.add_mutually_exclusive_group(required=False)
    zip_group.add_argument(
        "--zip",
        action="store_true",
        help="Zip directories before upload (default for directories)",
    )
    zip_group.add_argument(
        "--no-zip",
        action="store_true",
        help="Upload directories as a tree (no zip)",
    )

    parser.add_argument(
        "--zip-name",
        default=None,
        help="Zip output name when zipping a directory (default: <dir_name>.zip)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file(s) in Dropbox if present",
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


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    # Keep the directory name as top-level inside the zip.
    base = source_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_dir():
                continue
            rel = file_path.relative_to(base)
            zf.write(file_path, arcname=str(rel))


def _dropbox_upload_file(
    *, source_file: Path, token: str, dest_path: str, overwrite: bool
) -> None:
    try:
        import dropbox  # imported lazily so --help works even before deps
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Dropbox SDK not installed. Run `uv sync` (or install the 'dropbox' package)."
        ) from e

    dbx = dropbox.Dropbox(token)

    mode = (
        dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
    )

    with source_file.open("rb") as f:
        data = f.read()

    # Single-call upload is fine for typical artifact sizes. If you expect multi-GB
    # files, switch to upload sessions.
    dbx.files_upload(data, dest_path, mode=mode)


def _dropbox_upload_directory_tree(
    *,
    source_dir: Path,
    token: str,
    dest_dir: str,
    overwrite: bool,
) -> None:
    try:
        import dropbox  # imported lazily so --help works even before deps
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Dropbox SDK not installed. Run `uv sync` (or install the 'dropbox' package)."
        ) from e

    dbx = dropbox.Dropbox(token)
    mode = (
        dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
    )

    dest_dir = dest_dir.rstrip("/")

    for file_path in sorted(source_dir.rglob("*")):
        if file_path.is_dir():
            continue
        rel = file_path.relative_to(source_dir)
        dropbox_path = f"{dest_dir}/{rel.as_posix()}"
        with file_path.open("rb") as f:
            data = f.read()
        dbx.files_upload(data, dropbox_path, mode=mode)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    src = args.src or args.run
    if not src:
        raise ValueError("One of --src/--run must be provided")

    source = Path(src).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    dest = _assert_dropbox_path(str(args.dest))

    # Default behavior:
    # - For files: upload the file into dest folder
    # - For dirs: zip + upload into dest folder
    zip_dir = False
    if source.is_dir():
        zip_dir = True
        if args.no_zip:
            zip_dir = False
        elif args.zip:
            zip_dir = True

    token = None
    if not args.dry_run:
        token = _load_token(args)

    spec = UploadSpec(
        source=source,
        dropbox_dest=dest,
        zip_dir=zip_dir,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    print("Local source:", str(spec.source))

    # File upload
    if spec.source.is_file():
        dropbox_file_path = f"{spec.dropbox_dest.rstrip('/')}/{spec.source.name}"
        print("Dropbox file:", dropbox_file_path)

        sha = _sha256_file(spec.source)
        size_mb = spec.source.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        print(f"SHA256: {sha}")

        if spec.dry_run:
            print("Dry-run: not uploading.")
            return 0

        assert token is not None
        _dropbox_upload_file(
            source_file=spec.source,
            token=token,
            dest_path=dropbox_file_path,
            overwrite=spec.overwrite,
        )
        print("Upload complete.")
        return 0

    # Directory upload
    if spec.zip_dir:
        zip_name = args.zip_name or f"{spec.source.name}.zip"
        if not zip_name.endswith(".zip"):
            zip_name += ".zip"

        dropbox_zip_path = f"{spec.dropbox_dest.rstrip('/')}/{zip_name}"
        print("Dropbox file:", dropbox_zip_path)

        with tempfile.TemporaryDirectory(prefix="gw_dropbox_") as tmp:
            zip_path = Path(tmp) / zip_name
            _zip_directory(spec.source, zip_path)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            sha = _sha256_file(zip_path)
            print(f"Prepared zip: {zip_path.name} ({size_mb:.2f} MB)")
            print(f"SHA256: {sha}")

            if spec.dry_run:
                print("Dry-run: not uploading.")
                return 0

            assert token is not None
            _dropbox_upload_file(
                source_file=zip_path,
                token=token,
                dest_path=dropbox_zip_path,
                overwrite=spec.overwrite,
            )
            print("Upload complete.")
            return 0

    # Directory tree upload (no zip)
    dropbox_dir = spec.dropbox_dest
    print("Dropbox directory:", dropbox_dir)

    # Minimal preview in dry-run
    if spec.dry_run:
        files = [p for p in spec.source.rglob("*") if p.is_file()]
        print(f"Dry-run: would upload {len(files)} files.")
        return 0

    assert token is not None
    _dropbox_upload_directory_tree(
        source_dir=spec.source,
        token=token,
        dest_dir=dropbox_dir,
        overwrite=spec.overwrite,
    )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
