import tarfile
import os
import argparse
from pathlib import Path


def tarize(source_path: Path, output_filename: str):
    if not output_filename.endswith(".tar.gz"):
        output_filename += ".tar.gz"

    print(f"Compressing {source_path} into {output_filename}...")

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_path, arcname=source_path.name)

    print(f"Successfully created {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress a directory into a .tar.gz file."
    )
    parser.add_argument("--dir", required=True, help="Directory to compress")
    parser.add_argument(
        "--out",
        help="Output filename (without .tar.gz extension). Required if --each is not set.",
    )
    parser.add_argument(
        "--each", action="store_true", help="Compress each child folder separately"
    )

    args = parser.parse_args()

    source_path = Path(args.dir)
    if not source_path.exists():
        print(f"Error: Directory '{args.dir}' does not exist.")
        exit(1)

    if args.each:
        for child in source_path.iterdir():
            if child.is_dir():
                tarize(child, str(child))
    else:
        if not args.out:
            print("Error: --out is required when --each is not set.")
            exit(1)
        tarize(source_path, args.out)
