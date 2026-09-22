# Typical

Typical turns a pretrained language model into a direct probabilistic decision engine. Give it a state once —
a support ticket, a policy document, an agent trace, any block of text — and it answers many independent,
runtime-defined questions against that state in parallel, as calibrated probability distributions, with an
explicit "none of the above." No generation, no JSON parsing to fix up, no re-encoding the state per question.

Three typed primitives cover the decisions software actually needs to make:

- **Choice** — pick one of K runtime-defined options (or abstain).
- **Noul** — yes/no, read from a dedicated Bernoulli head, exactly invariant to which label you call "yes."
- **Score** — an ordinal level (urgency, severity, priority), trained with ordinal-smoothed targets so the
  probability mass concentrates near the true level instead of spreading uniformly over "wrong."

The state is encoded once into a KV cache; every question is a short causal suffix scored against that cache, so
adding more questions to a state is cheap — a single decision costs 45–60 ms depending on model size, and that
cost barely moves as the candidate set K grows from 2 to 256 (REPORT.md §3ab).

## The models

| model | backbone | JevBench standard / hard (public subset) | link |
|---|---|---|---|
| `typical-small-preview` | Qwen3-1.7B-Base | .750 / .387 | https://huggingface.co/OzLabs/typical-small-preview |
| `typical-small` | Qwen3-1.7B-Base | .694 / .432 | https://huggingface.co/OzLabs/typical-small |
| `typical-medium` | Qwen3-4B-Base | .806 / .423 | https://huggingface.co/OzLabs/typical-medium |

`typical-small` and `typical-medium` are Release 1: DecisionMix v2's hard curriculum, ordinal-smoothed Score,
a per-row Bernoulli Noul head, and 1,024-token decision states. `typical-small-preview` is the earlier Phase-6A
checkpoint, kept public as the reference point Release 1 is measured against. All three JevBench numbers are a
public-subset run (72 standard / 48 easy / 111 hard ids; not a ranked leaderboard entry — see each model's card).

`typical-medium` is the capability-per-millisecond knee of the ladder we've measured: +11 points of JevBench
standard and +11 points of MMLU-Pro among-K over `typical-small` for 1.25× the per-decision latency
(56–58 ms vs 45–46 ms at K = 2–32, REPORT.md §3ab/§3af). A 14B candidate (`tl1b`) reaches JevBench standard
**.931** — the best number this project has produced — but is not yet released; see `PROJECT.md` §1 and §7 for
why and what's still open.

## Quickstart

```bash
pip install -r inference/requirements.txt
```

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")  # or "typical-medium" / "typical-small-preview"

# K-way choice over a fixed label set -> {label: p, ...} + p_null
m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])

# Yes/no -> P(yes)
m.noul(state, "Is the order still under warranty?")

# Ordinal levels -> {level: p, ...} + p_null + expected (E[index])
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])
```

`state` is a string or a JSON-serialisable dict. Runnable end to end: `uv run --no-sync python inference/example.py`
(`inference/example.py`). The `inference/` package (`inference/README.md`) is self-contained — no dependency on
this training repo, just `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy` — and is numerically
verified against the internal decider (`inference/test_parity.py`, max abs probability diff `0.0` on CPU and MPS).

## Demo

```bash
uv run --no-sync python demo/app.py
```

Opens a local Gradio app at `http://127.0.0.1:7860`: a playground (one state, one question, see the probability
bar chart and latency), a batch view (one state, several questions, all scored against a single KV-encode of the
state — this is what "cached state" buys you), and the release-page tables reproduced from `releases/*.md` and
`REPORT.md` §3ab. See `demo/README.md`.

## Where the docs live

- `PROJECT.md` — start here: document map, code map, run registry, findings ledger, ops rules, open questions.
- `REPORT.md` — the consolidated results log, in chronological sections (§3a, §3b, … §3ah); every claim has a
  matched baseline and a section number.
- `PLAN7.md` — the current roadmap (scaling ladder, mixture sweep, typed primitives, DecisionMix v2, calibration,
  large-K path) and its execution notes.
- `RESULTS.md` — one results table per model family/size, pulled straight from run artefacts.
- `COMPARE.md` — the competitive picture against the closed Jev/System One family and the open JevBench leaderboard.
- `NOVELTY.md` — the contribution story: what is and isn't novel, and why.
- `releases/*.md` — one card per public release: exact backbone/adaptation/readout, training args, full results
  tables, known limitations, license.

## How to train

Training code is not yet public (the release cards ship inference only). The Release-1 recipe at 1.7B
(`typical-small`, checkpoint `ts1b`, from `releases/typical-small.md`):

```bash
uv run --no-sync python train.py --readout native --nc_head n3 --nc_render letters_nonull --null factored \
    --tap_layer 20 --zscore --lora_r 16 --lora_layers 8 \
    --data data_v5 --extra_data data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u \
    --bucket_map data_wf_long=W,data_wh=W,data_u=U \
    --family_weights E:0.40,K:0.15,W:0.35,U:0.10 --null_aug W:0.20 \
    --ordinal_smooth 0.7 --noul_head bern --max_state 1024 \
    --bs 64 --grad_accum 4 --eval_cap 1500 --steps 12000 \
    --ckpt_upload --wandb --hf_repo guychuk/pcdm-runs
```

The 4B (`typical-medium`, `tm1b`) is the same recipe on `Qwen/Qwen3-4B-Base` at tap 26/36 — see
`releases/typical-medium.md` for its exact args. `PROJECT.md` §6 has the current pod-ops rules (torch/kernel
pinning, micro-batch sizing, watcher rules) for anyone reproducing a run on rented GPUs.

## License and data notes

- Backbones (`Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`): Apache-2.0.
- Training data is a mix of public NLU/NLI sets, a distilled knowledge-MCQ corpus, an in-repo rubric-conditioned
  workflow generator, three HF-sourced workflow datasets (MIT / Apache-2.0), and DecisionMix v2 (`data_wh`, a
  programmatic in-repo rule engine with no external license constraints; `data_u`, whose UNLI portion is MIT and
  whose `metaeval/ambient` / `metaeval/chaos-mnli-ambiguity` portions do not declare a license on their HF cards —
  flagged, not asserted). Full per-source breakdown and licenses: `releases/typical-small.md`'s Data/License
  sections.
- JevBench numbers throughout this repo are a **public-subset run** against `fstandhartinger/jevbench` v1.2.1 (72
  standard / 48 easy / 111 hard public ids) — not a submitted or ranked leaderboard entry. The 72 MIT-licensed
  original JevBench items are the only public JevBench material that is training-eligible, and none of it was
  trained on for any released checkpoint.
