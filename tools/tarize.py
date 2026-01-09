import tarfile
import os
import argparse
from pathlib import Path


def tarize(source_dir: str, output_filename: str):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' does not exist.")
        return

    if not output_filename.endswith(".tar.gz"):
        output_filename += ".tar.gz"

    print(f"Compressing {source_dir} into {output_filename}...")

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_path, arcname=source_path.name)

    print(f"Successfully created {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress a directory into a .tar.gz file."
    )
    parser.add_argument("--dir", required=True, help="Directory to compress")
    parser.add_argument(
        "--out", required=True, help="Output filename (without .tar.gz extension)"
    )

    args = parser.parse_args()
    tarize(args.dir, args.out)
