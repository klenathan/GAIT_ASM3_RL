# Uploading training runs to Proton Drive (no root, SSH-only)

This repo includes a tiny uploader wrapper around `rclone`:

- Script: `tools/gw_upload.py`
- Purpose: upload `./runs/...` to your Proton Drive remote
- Works on school Linux machines without root (only needs `python3` + `rclone`)

## 0) Prerequisites

- The school machine has:
  - `rclone` already installed (you said it is)
  - `python3` available
- You can SSH to the school machine.
- You have some other machine (your laptop/PC) with:
  - a browser (to login to Proton)
  - `rclone` installed (recommended)

## 1) Create the Proton Drive remote on your personal machine

On your personal machine (NOT the school machine):

1. Run interactive config:

```
rclone config
```

2. Choose:
- `n` (New remote)
- name it (example): `proton`
- pick the **Proton Drive** backend (the menu will show a number)
- follow prompts to login

3. Confirm the remote works:

```
rclone lsd proton:
```

This should list folders in your Proton Drive.

## 2) Copy the rclone config to the school machine (SSH-only)

`rclone` stores remotes in a config file. You can create it on your personal
machine (with browser auth) and then copy it to the school machine.

### 2.1 Find your rclone config file path (personal machine)

```
rclone config file
```

You should see something like:
- Linux: `~/.config/rclone/rclone.conf`
- macOS: `~/.config/rclone/rclone.conf` (often the same)
- Windows: shown by the command

### 2.2 Copy the config file to the school machine

From your personal machine:

```
scp ~/.config/rclone/rclone.conf <user>@<school-host>:~/.config/rclone/rclone.conf
```

If `~/.config/rclone/` does not exist on the school machine:

```
ssh <user>@<school-host> "mkdir -p ~/.config/rclone"
scp ~/.config/rclone/rclone.conf <user>@<school-host>:~/.config/rclone/rclone.conf
```

### 2.3 Verify on the school machine

SSH to the school machine and run:

```
rclone listremotes
rclone lsd proton:
```

If that works, you’re authenticated.

## 3) Upload runs from the school machine

### Upload all GridWorld runs

From repo root:

```
python3 tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld
```

This uploads:
- local: `./runs/gridworld/`
- to: `proton:school/asm3/runs/gridworld/`

### Upload one specific run folder

```
python3 tools/gw_upload.py --remote proton --dest "school/asm3" --run runs/gridworld/<run_name>
```

This uploads to:
- `proton:school/asm3/runs/gridworld/<run_name>/`

### Dry-run first (recommended)

```
python3 tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld --dry-run
```

### Copy vs sync

- Default is `--mode copy` (safe): it uploads new/changed files, and **does not delete** anything on Proton.
- `--mode sync` (dangerous): makes the remote match local; **deletes** remote files not present locally.

Example sync:

```
python3 tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld --mode sync
```

## 4) Troubleshooting

### "rclone not found"
- Ensure `rclone` is on PATH on the school machine.

### Proton remote name differs
- If you named your remote something else in `rclone config`, pass that name to `--remote`.

### Permission issues writing config
- Use your home directory `~/.config/rclone/rclone.conf` (no root needed).

### Bandwidth limiting

You can pass extra rclone args using `--extra`:

```
python3 tools/gw_upload.py --remote proton --dest "school/asm3" --what gridworld --extra "--bwlimit=8M"
```

## 5) Notes

- The uploader is intentionally generic: it just wraps `rclone copy/sync`.
- It prints the exact `rclone` command it runs, so you can copy-paste it if needed.
