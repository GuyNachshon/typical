# typical-ai

**Typed, direct decisions from a pretrained language model.** State + question + a label set you
define at call time go in; a probability distribution over exactly those labels, plus an explicit
abstain, comes out. One forward pass, nothing generated, nothing to parse.

A self-contained decision head for `OzLabs/typical-small-preview`, `OzLabs/typical-small`, and
`OzLabs/typical-medium`. No training-repo dependencies (no `bench`/`train`/`data`/
`metrics`) -- just `torch`, `transformers`, `huggingface_hub`, `safetensors`, `numpy`.

## Install

```bash
pip install typical-ai
```

The import name is `typical_ai`, not `typical`: PyPI's `typical` is an unrelated, established
package, so taking that import name would break anyone who has both installed.

```python
from typical_ai import Typical

m = Typical.from_pretrained("OzLabs/typical-small")        # or OzLabs/typical-medium

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
m.noul(state,   "Is the order still under warranty?")
m.score(state,  "How urgent is this ticket?", ["0", "1", "2", "3"])
```

Models and their cards, including the measured trade-offs of each release, are at
<https://huggingface.co/OzLabs>. Source and the full experimental record:
<https://github.com/GuyNachshon/typical>.

(Or just copy the `typical/` directory next to your code -- it's a plain Python package,
no build step.)

## Usage

```python
from typical_ai import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")  # or "typical-small-preview" / "typical-medium"

# K-way choice over a fixed label set -> {label: p, ...} + p_null
m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])

# Yes/no -> P(yes)
m.noul(state, "Is the order still under warranty?")

# Ordinal levels -> {level: p, ...} + p_null + expected (E[index] under the candidates)
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])

# Full JevBench-style decide() -- mirrors pcdm_jev.decider.PCDMDecider.decide exactly,
# including the runtime block (latency, p_null, truncation flags).
probs, runtime = m.decide(
    state,
    {"type": "choice", "instructions": "What does the customer want?",
     "criteria": {"refund": "money back", "replacement": "a new item shipped"}},
    ["refund", "replacement"],
)
```

`state` is either a string or a JSON-serialisable dict (dicts are dumped with
`json.dumps` before tokenising, same as the original decider).

See `example.py` for a runnable end-to-end script.

## What's inside

`typical/` is a trimmed, numerically-identical port of this repo's native-readout serving
path (`pcdm_jev.decider.PCDMDecider(mode="native")`):

- `typical/backbone.py` -- frozen Qwen3 trunk truncated at `tap_layer`, LoRA on its top
  `lora_layers` blocks (`pcdm/encode.py`'s `Backbone`, minus the training-only
  FeatureCache/EmbedEncoder/forward() machinery the serving path never touches).
- `typical/native.py` -- `NativeHead` (factored null, the per-row Bernoulli "noul" route,
  the `letters`/`tags`/`letters_nonull`/`query_only` renderers) and `native_kv_decide`
  (encode the state once into a KV cache, score every query's suffix against it).
- `typical/core.py` -- `Typical.from_pretrained` (downloads `best.pt`, reads
  `checkpoint["args"]` for backbone id / `tap_layer` / `lora_r` / `lora_layers` /
  `nc_head` / `nc_render` / `null` / `noul_head` / `score_head` to rebuild the exact
  backbone+head shape, then loads the LoRA + tower weights) plus the `choice`/`noul`/
  `score`/`decide` API and `query_text`/`to_labels` (verbatim from `pcdm_jev/decider.py`).

Not ported: `nc_head` values `n2`/`n2n3` (Qwen3-Embedding candidate vectors) -- neither
shipped checkpoint trains with them; `native_kv_decide` raises `NotImplementedError` if a
future checkpoint needs that path. z-score calibration (`--zscore`) needs no special
handling -- its buffers live in the checkpoint's `tower` state_dict like any other weight.

## Verification

`inference/test_parity.py` (gated behind `RUN_SLOW=1`, downloads the real 1.7B checkpoint)
runs 6 items through both the original `PCDMDecider(mode="native")` and `Typical` on the
same device and checkpoint, and asserts identical probabilities (max abs diff was `0.0`
against `guychuk/pcdm-runs/typical-small/best.pt` on both CPU and MPS), plus the noul
reversed-label control (`["no","yes"]` vs `["yes","no"]` give the same P(yes) through the
Bernoulli head) and a check that `typical-small-preview`'s `noul_head="choice"` checkpoint
still answers `noul()` correctly through the plain K-way path.

```bash
RUN_SLOW=1 python inference/test_parity.py
```
