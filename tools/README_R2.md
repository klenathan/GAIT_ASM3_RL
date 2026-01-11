# Cloudflare R2 Upload Tool

This tool allows you to easily upload files to a Cloudflare R2 bucket using a Python script.

## Prerequisites

Ensure you have the necessary dependencies installed. The project uses `uv` for dependency management.

The required packages (`boto3`, `python-dotenv`) should be added to your environment. If not, you can add them:

```bash
uv add boto3 python-dotenv
```

## Configuration

1.  **Get Credentials**: You need your Cloudflare R2 credentials:
    *   Account ID
    *   Access Key ID
    *   Secret Access Key
    *   Bucket Name

2.  **Create Environment File**:
    *   Copy the example configuration file:
        ```bash
        cp tools/example.r2.env .r2.env
        ```
    *   Edit `.r2.env` and fill in your actual credentials.

    **Note:** The script searches for `.r2.env` in the current working directory first, then in the project root.

## Usage

Use `uv run` to execute the script. You must specify the source file (`--src`) and the destination path in the bucket (`--dest`).

### Command Format

```bash
uv run python tools/r2_upload.py --src <path_to_local_file> --dest <path_in_bucket>
```

### Examples

**Upload a single file to the root of the bucket:**

```bash
uv run python tools/r2_upload.py --src data/model.pt --dest model.pt
```

**Upload a file to a folder within the bucket:**

```bash
uv run python tools/r2_upload.py --src runs/experiment_1/log.txt --dest experiments/2023-10-27/log.txt
```

## Troubleshooting

*   **Error: .r2.env file not found**: Make sure you have created the `.r2.env` file in the directory where you are running the command, or in the project root.
*   **Upload failed**: Check your internet connection and ensure your credentials in `.r2.env` are correct and have write permissions to the bucket.
