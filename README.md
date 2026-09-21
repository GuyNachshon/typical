# Typical — website

Research + marketing site for the Typical decision models (`site/`), plus a tiny local API
so the demos run against the real model.

## Static (replays, no model)

```bash
python3 -m http.server 8788 -d site      # http://localhost:8788
```

Demos resolve from `site/data/replays.json`; the hero snake plays `site/data/snake-replay.json`.

## Live (real model on this machine)

```bash
uv sync
uv run uvicorn server:app --port 8787    # http://localhost:8787
```

First request downloads `OzLabs/typical-small` (LoRA + head, tens of MB; the Qwen3 backbone
comes from the `transformers` cache). The nav pill flips to `LIVE`; every result shows its
measured latency and device.

## Regenerate data

```bash
uv run precompute.py                     # site/data/{models,latency,reliability,chance}.json  <- runs/, releases/
uv run scripts/precompute_research.py    # site/data/research-*.json
uv run scripts/record_replays.py         # site/data/replays.json   (needs the live server)
node scripts/record_snake.mjs            # site/data/snake-replay.json (needs the live server)
```

`runs/`, `releases/` and `inference/` are checked out verbatim from branch
`gpu-runpod-full-experiment`; every number on the site traces to a file there.
