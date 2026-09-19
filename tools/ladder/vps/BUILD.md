# CPU build for the ladder VPS

No CUDA, no node. Sizing (measured 2026-09-19 on an 8-core/16-thread desktop CPU at 8 torch threads): mantis
≈ 9.3 s per compound turn at PUCT-256 (≈ 4.6 s per stone), strix ≈ 3.1 s per turn at 256 sims (≈ 50 ms when
its root VCF solver decides). A 4-vCPU VPS reads roughly twice that; an `unlimited` or `turn:60000`+ time
control keeps both inside the clock.

```sh
# 1. the repo and the engine wheel (rustup provisions the pinned toolchain from rust-toolchain.toml)
git clone <repo> /opt/mantis && cd /opt/mantis
uv sync                                   # CPU torch is the default dependency group; builds mantis._engine

# 2. strix: the vendored tree, its own venv (CPU torch) and hexo_rs, then the pinned checkpoint
make vendor && bash tools/vendor_build_strix.sh
cp <checkpoint_00237000.pt> vendor/external/strix_models/   # sha256 verified against vendor/pins.toml at every load

# 3. mantis: the checkpoint to play, named as its stamp names it (run-id + step + content hash); never in git
mkdir -p checkpoints/<run> && cp <run>_<step>_<hash>.ckpt checkpoints/<run>/

# 4. the units, one env file per bot, receipts under /var/lib/ladder/<instance>/receipts/<net_hash8>/
sudo install -m 0644 tools/ladder/vps/ladder-bot@.service /etc/systemd/system/
sudo install -d -m 0700 /etc/ladder && sudo install -d -o mantis /var/lib/ladder
sudo systemctl daemon-reload && sudo systemctl enable --now ladder-bot@strix ladder-bot@mantis
```

Smoke before enabling: `HEXO_TOKEN=... .venv/bin/python tools/ladder_bot.py --backend strix --server $HEXO_SERVER
--work-dir /tmp/ladder-smoke --no-reconnect` must print `stream open` and the bot must show `online` on
`GET /api/bots?online=1`. The witness on any receipt: `tools/ladder_bot.py --backend <b> ... --replay <receipt.json>`.
