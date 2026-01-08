# Simple upload from school machine via `scp`

Goal: train on the school Linux machine, then copy the resulting run folder
straight to your own machine using SSH/SCP (no Proton auth on school).

This repo includes:

- Tool: `tools/gw_scp_push.py`
- Copies a **single** run directory like `runs/gridworld/<run_name>`

Once the data is on your own machine, you can run `rclone` to upload to Proton
Drive from there.

## 1) One-time SSH setup (recommended)

### Option A: SSH keys

On your host machine (your laptop/PC):

```
ssh-keygen -t ed25519
```

Copy your public key to the school machine (so you can SSH *into* school):

```
ssh-copy-id <you>@<school-host>
```

You also need the other direction: school machine -> host machine.

Copy your **host machine public key** onto the **host machine** `authorized_keys`,
so the school machine can SSH to it.

Simplest approach:

1) On the school machine, generate a key:

```
ssh-keygen -t ed25519
```

2) Copy the school machine **public key** to your host machine:

```
cat ~/.ssh/id_ed25519.pub
```

3) On your host machine, append it to `~/.ssh/authorized_keys`:

```
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Now the school machine should be able to do:

```
ssh <your-host-user>@<your-host>
```

### Option B: password SSH

This works too, but `scp` will prompt for a password every time.

## 2) Copy ONE run folder from school -> host

From the **school machine**, inside this repo:

```
python3 tools/gw_scp_push.py \
  --run runs/gridworld/<run_name> \
  --host <your-host-user>@<your-host> \
  --dest "~/asm3_uploads/GAIT_ASM3_RL"
```

Result on your host machine:

- `~/asm3_uploads/GAIT_ASM3_RL/<run_name>/...`

### Dry run

```
python3 tools/gw_scp_push.py --run runs/gridworld/<run_name> --host <user>@<host> --dest "~/asm3_uploads" --dry-run
```

### Non-standard SSH port / identity key

```
python3 tools/gw_scp_push.py \
  --run runs/gridworld/<run_name> \
  --host <user>@<host> \
  --dest "~/asm3_uploads/GAIT_ASM3_RL" \
  --port 2222 \
  --identity ~/.ssh/id_ed25519 \
  --compress
```

## 3) Upload to Proton Drive from your host machine

Once the folder is on your host machine, upload it however you want.
If you already have an rclone remote (example `proton`):

```
rclone copy ~/asm3_uploads/GAIT_ASM3_RL/<run_name> proton:school/asm3/runs/gridworld/<run_name>
```

## Troubleshooting

- "Permission denied (publickey)": your host machine isn’t accepting the
  school machine key. Ensure the school machine public key is in
  `~/.ssh/authorized_keys` on the host.
- "Connection timed out": your host might be behind NAT/firewall. Consider
  using Tailscale/WireGuard, or reverse the direction (pull from host).
