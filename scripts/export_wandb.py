"""Export training curves from W&B project `pcdm` into site/data/curves/<run>.json (+ index.json).

Schema per run: {"run": name, "state", "config": {backbone, steps, bs, lora_r, tap_layer, readout, ...},
  "series": {"train/loss": [[step, value], ...], "val/nll": [...], ...}}  downsampled to <= 400 points.
Reads WANDB_API_KEY from .env. Usage: uv run --with wandb scripts/export_wandb.py
"""
import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data" / "curves"
OUT.mkdir(parents=True, exist_ok=True)

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

import wandb  # noqa: E402

api = wandb.Api(timeout=120)
try:
    runs = list(api.runs("pcdm", per_page=200))
except Exception:
    runs = list(api.runs(f"{api.default_entity}/pcdm", per_page=200))

KEEP_CFG = ["backbone", "steps", "bs", "grad_accum", "lora_r", "lora_layers", "tap_layer", "readout", "nc_head", "null",
            "noul_head", "score_head", "data", "extra_data", "family_weights", "max_state", "seed", "lr", "lora_lr"]


def downsample(points, n=400):
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


index = []
for r in runs:
    series = {}
    for row in r.scan_history(page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        for k, v in row.items():
            if k.startswith("_") or not isinstance(v, (int, float)):
                continue
            if k.startswith("train/") or k.startswith("val") or k.startswith("eval/"):
                series.setdefault(k, []).append([step, round(float(v), 6)])
    if not series:
        continue
    series = {k: downsample(v) for k, v in series.items()}
    cfg = {k: r.config.get(k) for k in KEEP_CFG if k in r.config}
    doc = {"run": r.name, "id": r.id, "state": r.state, "created": str(r.created_at), "config": cfg, "series": series,
           "source": f"wandb:pcdm/{r.id}"}
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", r.name)
    (OUT / f"{safe}.json").write_text(json.dumps(doc, separators=(",", ":")))
    index.append({"run": r.name, "file": f"{safe}.json", "state": r.state, "created": str(r.created_at),
                  "backbone": cfg.get("backbone"), "steps": cfg.get("steps"), "keys": sorted(k for k in series if k == "train/loss" or k.startswith("val") or k.startswith("eval/"))[:12],
                  "final_loss": series.get("train/loss", [[None, None]])[-1][1]})
    print(f"{r.name:32s} {len(series)} keys  loss points {len(series.get('train/loss', []))}")

index.sort(key=lambda x: x["created"])
(OUT / "index.json").write_text(json.dumps({"project": "pcdm", "runs": index, "note": "downsampled to <= 400 points per series; source per run in each file"}, indent=1))
print("runs exported:", len(index))
