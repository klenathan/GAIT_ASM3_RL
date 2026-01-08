# Uploading training runs to Dropbox with rclone (token paste workflow)

This repo includes a Dropbox-focused uploader wrapper:

- Tool: `tools/gw_rclone_dropbox.py`

Why: on an SSH-only school machine, you usually can’t open a browser to login.
So you do the browser OAuth on your host machine and paste the token back.

## 0) Prerequisites

- School machine: `python3` + `rclone` available
- Host machine (your laptop/PC): `rclone` available + browser

## 1) Create a Dropbox remote on the school machine using `rclone authorize`

### 1.1 On your host machine (with browser)

Run:

```
rclone authorize "dropbox"
```

This will:
- open a browser window
- have you login to Dropbox and approve access
- then print a token block that looks like JSON

Copy the ENTIRE output token block (including braces), for example:

```
{"access_token":"...","token_type":"bearer",...}
```

### 1.2 On the school machine (SSH session)

Run:

```
rclone config
```

Then follow the prompts:

1) `n` (New remote)
2) Name it (example): `dropbox`
3) Storage type: choose `Dropbox`
4) rclone will ask something like:

- "Use auto config?" → choose `n` (NO) because you can’t open a browser
- then it will prompt for a token / config JSON (commonly shown as `config_token>`)

When you see the token prompt (e.g. `config_token>`):

- Paste the JSON token you copied from `rclone authorize "dropbox"`
- Press Enter

Finish the config and quit.

### 1.3 Verify the remote works on the school machine

```
rclone listremotes
rclone lsd dropbox:
```

If `rclone lsd dropbox:` lists folders, auth is done.

## 2) Upload runs from the school machine

### Upload one specific run folder

```
python3 tools/gw_rclone_dropbox.py \
  --remote dropbox \
  --dest "asm3" \
  --run runs/gridworld/<run_name>
```

This uploads to:
- `dropbox:asm3/runs/gridworld/<run_name>/...`

### Upload all GridWorld runs

```
python3 tools/gw_rclone_dropbox.py --remote dropbox --dest "asm3" --what gridworld
```

### Dry-run first (recommended)

```
python3 tools/gw_rclone_dropbox.py --remote dropbox --dest "asm3" --what gridworld --dry-run
```

## 3) Notes

- Default mode is `copy` (safe). `--mode sync` will delete remote extras.
- If you already configured a Dropbox remote with a different name, pass it via `--remote`.
