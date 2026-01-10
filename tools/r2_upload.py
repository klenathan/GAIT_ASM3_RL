import os
import argparse
import sys
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pathlib import Path


def setup_client():
    """
    Sets up the boto3 client for Cloudflare R2 using credentials from .r2.env
    """
    # Look for .r2.env in the current directory or the same directory as the script
    env_path = Path(".r2.env")
    if not env_path.exists():
        # Try finding it in the project root if running from tools/
        env_path = Path(__file__).parent.parent / ".r2.env"

    if not env_path.exists():
        print(
            "Error: .r2.env file not found. Please copy tools/example.r2.env to .r2.env and fill in your credentials."
        )
        sys.exit(1)

    load_dotenv(dotenv_path=env_path)

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")

    if not all([account_id, access_key_id, secret_access_key, bucket_name]):
        print("Error: Missing required environment variables in .r2.env")
        print(
            "Required: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
        )
        sys.exit(1)

    # Cloudflare R2 S3 API endpoint
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3_client = boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    return s3_client, bucket_name


def upload_file(src_path, dest_path):
    """
    Uploads a file to Cloudflare R2
    """
    s3_client, bucket_name = setup_client()

    src_file = Path(src_path)
    if not src_file.exists():
        print(f"Error: Source file '{src_path}' does not exist.")
        sys.exit(1)

    print(f"Uploading '{src_path}' to R2 bucket '{bucket_name}' at '{dest_path}'...")

    try:
        s3_client.upload_file(str(src_file), bucket_name, dest_path)
        print("Upload successful!")
    except ClientError as e:
        print(f"Upload failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload a file to Cloudflare R2.")
    parser.add_argument(
        "--src", required=True, help="Path to the source file to upload"
    )
    parser.add_argument(
        "--dest", required=True, help="Destination path (key) in the R2 bucket"
    )

    args = parser.parse_args()

    upload_file(args.src, args.dest)


if __name__ == "__main__":
    main()
