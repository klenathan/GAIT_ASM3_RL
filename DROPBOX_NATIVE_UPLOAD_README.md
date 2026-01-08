# Upload to Dropbox without rclone (native Dropbox API)

This repo includes a Dropbox uploader that does **not** use `rclone`.

- Tool: `tools/gw_dropbox_upload.py`
- Upload method: zip the run directory, then upload the zip via Dropbox API
- Auth: long-lived access token (simplest)

## 0) One-time: create a Dropbox access token (on your host machine)

1) Open the Dropbox developer console:
- https://www.dropbox.com/developers/apps

2) Create an app:
- Choose: "Scoped access"
- Choose: "Full Dropbox" or "App folder" (recommended: App folder)

3) Permissions (scopes):
- Enable at least file write access (Dropbox UI typically calls these
  `files.content.write`)

4) Generate an access token:
- In the app settings page, find "Generated access token"
- Click generate and copy it

Keep it secret.

## 1) On the school machine: set the token (where to paste)

SSH to the school machine, go to the repo root, then set the token in your
shell environment:

```
export DROPBOX_ACCESS_TOKEN="PASTE_THE_TOKEN_HERE"
```

If you want it to persist for future sessions, add it to `~/.bashrc` or
`~/.zshrc` (depending on your shell) and re-login.

Alternative: store it in a file readable only by you:

```
mkdir -p ~/.secrets
chmod 700 ~/.secrets
nano ~/.secrets/dropbox_token.txt
chmod 600 ~/.secrets/dropbox_token.txt
```

Then the tool can use:

```
--token-file ~/.secrets/dropbox_token.txt
```

## 2) Install deps with uv (no root)

Since this repo uses `uv` and the dependency is in `pyproject.toml`, on the
school machine just run:

```
uv sync
```

## 3) Upload a single run folder

Example uploads `runs/gridworld/<run_name>` to Dropbox folder `/asm3/runs/gridworld`
(as a zip file named `<run_name>.zip`).

```
uv run python tools/gw_dropbox_upload.py \
  --run runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld"
```

### Dry run

```
uv run python tools/gw_dropbox_upload.py \
  --run runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld" \
  --dry-run
```

### Overwrite existing zip

By default Dropbox will create a new name if the file exists. To overwrite:

```
uv run python tools/gw_dropbox_upload.py \
  --run runs/gridworld/<run_name> \
  --dest "/asm3/runs/gridworld" \
  --overwrite
```

## Notes / limits

- This tool currently uploads the zip in a single API call. If your run folders
  become very large (hundreds of MB+), we can switch to a chunked upload
  session.
