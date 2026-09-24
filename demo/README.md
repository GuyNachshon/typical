# Typical demo

A local Gradio app that runs `OzLabs/typical-small` / `typical-medium` / `typical-small-preview`
on this machine and doubles as the release page (launch copy, model family table, latency ladder).

## Run

```bash
uv run --no-sync python demo/app.py
```

Open http://127.0.0.1:7860.

## What it does

- **Header** -- the launch copy: what Typical is, the three primitives (Choice / Noul / Score),
  model links, latency headline.
- **Model selector** -- pick a model; it lazy-loads on first use (or when you change the dropdown)
  and stays cached until you switch, so only one model is ever resident.
- **Playground** -- one state, one question, run Choice / Score / Noul, see a bar chart of the
  probability distribution (including &#8709;), the argmax, `p_null`, and wall-clock latency. Four
  preloaded examples, including one designed to abstain (the state contradicts the question's
  premise).
- **Batch** -- one state, several `type | question | labels` lines, all scored against a single
  KV-encode of the state (`typical.native.native_kv_decide`) -- this is what "cached state" buys you:
  the state is read once, every query only pays for its own short suffix.
- **Results** -- static: the model family table and JevBench numbers from `releases/typical-*.md`,
  the latency ladder from REPORT.md &sect;3ab, and the JevBench disclosure.

## Model download

First use of a model downloads its checkpoint (`best.pt`) from Hugging Face Hub via
`huggingface_hub` -- backbone weights are not re-downloaded per model (`Qwen/Qwen3-1.7B-Base` /
`Qwen3-4B-Base` come from the `transformers` cache the first time any typical-* model needs them);
the LoRA + tower state in `best.pt` itself is on the order of tens of MB. If `HF_TOKEN` is needed
for a gated/private repo, this app reads it from the project's `.env` (never printed).

## Device notes

- **MPS (Apple Silicon):** `Typical.from_pretrained(..., device="auto")` picks `mps` automatically.
  This is what the demo was built and verified against.
- **CPU:** works, just slower per decision (still fine for the Playground tab; the Batch tab's
  single-KV-encode design keeps the per-query cost down regardless of device).
- **CUDA:** picked automatically if present; not exercised for this demo.
