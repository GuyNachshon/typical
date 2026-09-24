"""Typical demo -- a single Gradio app that doubles as the release page.

Run: uv run --no-sync python demo/app.py  ->  http://127.0.0.1:7860

Loads the public `inference/typical` package directly (no pip install, no copy) and runs
whichever OzLabs/typical-* model you pick, on this machine's best device (MPS on a Mac).
"""
import os
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

# ponytail: HF_TOKEN lives in .env, not the shell env. Load it by hand instead of adding
# python-dotenv as a dependency for three lines -- never printed, never logged.
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from typical import Typical  # noqa: E402  (needs sys.path patched first)
from typical.core import to_labels  # noqa: E402
from typical.native import native_kv_decide  # noqa: E402

MODELS = ["OzLabs/typical-small", "OzLabs/typical-medium", "OzLabs/typical-small-preview"]

HEADER = """
# Typical

Typical turns a frozen Qwen3 backbone plus a thin LoRA adapter into a **decision head**: give it a
state and a question, and it returns a calibrated probability distribution over *your* candidates
in a single forward pass -- plus a native "I don't know" (&#8709;). No generation, no parsing a
completion for an answer: the model scores whatever candidate set you hand it at inference time.

**Three primitives, one head:**
- **Choice** -- K-way question &#8594; `{label: probability}` + `p_null`
- **Noul** -- yes/no &#8594; `P(yes)`, through a dedicated Bernoulli head that's exactly order-invariant
- **Score** -- ordinal levels &#8594; `{level: probability}` + `p_null` + expected level (E[index])

One pass, one KV-encode of the state, calibrated probabilities and abstention included: 45 ms
(1.7B) / 56 ms (4B) for a single K=2 decision on an H100 (REPORT.md &sect;3ab -- your local latency
on MPS/CPU below will differ).

Models:
[OzLabs/typical-small-preview](https://huggingface.co/OzLabs/typical-small-preview) &middot;
[OzLabs/typical-small](https://huggingface.co/OzLabs/typical-small) &middot;
[OzLabs/typical-medium](https://huggingface.co/OzLabs/typical-medium)

Built by OzLabs.
"""

RESULTS_MD = """
## Model family (JevBench public-subset, REPORT.md &sect;3ae/&sect;3af)

| Model | Backbone | Tap | JevBench std / easy / hard | Brier std / hard | ECE std | Single-decision latency, K=2 (ms) |
|---|---|---|---|---|---|---|
| `typical-small-preview` | Qwen3-1.7B-Base | 20/28 | .750 / 1.00 / .387 | .40 / .88 | .15 | same 1.7B shape as `typical-small` |
| `typical-small` | Qwen3-1.7B-Base | 20/28 | .694 / 1.00 / .432 | .40 / .79 | .11 | 45 |
| `typical-medium` | Qwen3-4B-Base | 26/36 | .806 / 1.00 / .423 | .30 / .77 | .09 | 56 |

## Latency ladder (REPORT.md &sect;3ab, one H100, one torch build, state length 256)

| | single decision, K=2 / 32 / 256 (ms) | marginal per query, M=32, K=2 / 32 / 256 (ms) | peak memory, K=2&#8594;256 (GB) |
|---|---|---|---|
| 1.7B (`typical-small` recipe) | 45 / 46 / 106 | 2.7 / 3.7 / 28.0 | 6.4 &#8594; 9.3 |
| 4B (`typical-medium` recipe) | 56 / 56 / 118 | 3.3 / 6.1 / 50.6 | 14.6 &#8594; 18.6 |

A single decision costs ~45-56 ms regardless of most of K -- the KV-cached state and a short suffix
dominate; batching queries against one cached state (see the Batch tab) drops the marginal cost to a
few ms/query.

## JevBench disclosure

This is a **public-subset run against `fstandhartinger/jevbench` v1.2.1 (72 standard / 48 easy / 111
hard ids), not a ranked leaderboard entry** (the official leaderboard requires >=95% coverage
including 146 non-public judge items we don't have). Reported probabilities are the head's softmax
conditioned on non-&#8709; (P(&#8709;) dropped, rest renormalized; mean p_null &#8776; .03). Chance/majority
baseline on this split is .311 standard / .284 easy / .336 hard (n_eff = 36) -- `typical-small`'s hard
tier (.432) and `typical-medium`'s (.423) are within ~1 SE of chance; standard tier is not.
"""

# ---------------------------------------------------------------------------
# Model cache: exactly one Typical instance resident at a time.
# ---------------------------------------------------------------------------
_cache = {"repo": None, "model": None, "load_s": 0.0}


def get_model(repo: str):
    """Lazy-load `repo`, evicting whatever was cached before it. Returns (model, load_s)."""
    if _cache["repo"] != repo:
        t0 = time.perf_counter()
        _cache["model"] = Typical.from_pretrained(repo, device="auto")
        _cache["repo"] = repo
        _cache["load_s"] = time.perf_counter() - t0
    return _cache["model"], _cache["load_s"]


def load_status(repo: str) -> str:
    _, load_s = get_model(repo)
    dev = _cache["model"].device
    return f"**{repo}** loaded on `{dev}` in {load_s:.1f}s."


def _bar_df(probs: dict, p_null: float) -> pd.DataFrame:
    rows = [{"label": k, "probability": v} for k, v in probs.items()]
    rows.append({"label": "∅ (abstain)", "probability": p_null})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Playground tab
# ---------------------------------------------------------------------------
def toggle_labels(primitive: str):
    return gr.update(visible=(primitive != "Noul"))


def run_playground(repo: str, state: str, question: str, primitive: str, labels_str: str):
    if not state.strip() or not question.strip():
        raise gr.Error("State and Question are required.")
    model, load_s = get_model(repo)

    t0 = time.perf_counter()
    if primitive == "Noul":
        result = model.choice(state, question, ["no", "yes"])  # same route as model.noul(), keeps p_null
        expected_line = ""
    else:
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]
        if len(labels) < 2:
            raise gr.Error("Provide at least two comma-separated labels.")
        if primitive == "Choice":
            result = model.choice(state, question, labels)
            expected_line = ""
        else:  # Score
            result = model.score(state, question, labels)
            expected_line = f"**Expected level index:** {result['expected']:.2f}  \n"
    latency_s = time.perf_counter() - t0

    p_null = result.pop("p_null")
    result.pop("expected", None)
    probs = result
    argmax_label = max(probs, key=probs.get)

    summary = (
        f"**Argmax:** {argmax_label}  (p={probs[argmax_label]:.3f})  \n"
        f"**p_null (∅):** {p_null:.3f}  \n"
        f"{expected_line}"
        f"**Latency:** {latency_s * 1000:.1f} ms  \n"
        f"**Model load time (cached after first use):** {load_s:.1f}s on `{model.device}`"
    )
    return _bar_df(probs, p_null), summary


EXAMPLES = [
    # 1. Support-ticket routing, Choice, 4 teams.
    [
        "Ticket #7734: A customer writes: 'I was charged twice for my last order and the "
        "tracking number you sent doesn't work. I need this fixed today.'",
        "Which team should own this ticket?",
        "Choice",
        "billing, shipping, support, retention",
    ],
    # 2. Severity, Score 1-4.
    [
        "Incident report: the payment API returned HTTP 500 for 12 minutes, affecting "
        "roughly 3% of checkout attempts before auto-recovering. No data loss.",
        "How severe is this incident, where 1=minor, 2=moderate, 3=major, 4=critical?",
        "Score",
        "1, 2, 3, 4",
    ],
    # 3. Policy, Noul.
    [
        "Refund policy: refunds are available in cash within 30 days of purchase. After "
        "30 days, only store credit is issued, and only if the item is unused. Customer "
        "request: purchased 45 days ago, item unused, wants a cash refund.",
        "Is the customer eligible for a cash refund under this policy?",
        "Noul",
        "",
    ],
    # 4. Evidence/NLI-style Choice with an absent gold answer -- should abstain.
    [
        "Premise: the company's Q3 revenue grew 4% year-over-year, driven mainly by its "
        "cloud division.",
        "What was the primary driver of the decline in Q3 profit?",
        "Choice",
        "layoffs, currency headwinds, litigation costs",
    ],
]

# ---------------------------------------------------------------------------
# Batch tab -- one state, many queries, ONE KV-encode (native_kv_decide) so the
# cost of re-reading the state is paid exactly once, whatever K/M is.
# ---------------------------------------------------------------------------
def _parse_batch_line(line: str):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 2:
        raise gr.Error(f"Bad line (need 'type | question | labels'): {line!r}")
    typ = parts[0].lower()
    question = parts[1]
    labels = [l.strip() for l in parts[2].split(",") if l.strip()] if len(parts) > 2 else []
    if typ == "noul":
        labels = ["no", "yes"]
    elif len(labels) < 2:
        raise gr.Error(f"'{typ}' line needs >=2 comma-separated labels: {line!r}")
    return typ, question, labels


def run_batch(repo: str, state: str, questions_text: str):
    if not state.strip() or not questions_text.strip():
        raise gr.Error("State and at least one question line are required.")
    model, _ = get_model(repo)
    parsed = [_parse_batch_line(l) for l in questions_text.splitlines() if l.strip()]
    queries = [(q, labels) for _, q, labels in parsed]

    t0 = time.perf_counter()
    raws = native_kv_decide(model.head, model.model, state, queries, max_state=model.max_state)
    total_s = time.perf_counter() - t0

    rows = []
    for (typ, question, labels), raw in zip(parsed, raws):
        probs, p_null = to_labels(raw, labels)
        argmax_label = max(probs, key=probs.get)
        rows.append({
            "type": typ, "question": question, "argmax": argmax_label,
            "p": round(probs[argmax_label], 3), "p_null": round(p_null, 3),
        })
    summary = f"**{len(parsed)} queries against one cached state:** {total_s * 1000:.1f} ms total."
    return pd.DataFrame(rows), summary


BATCH_EXAMPLE_STATE = (
    "Ticket #7734: A customer writes: 'I was charged twice for my last order and the "
    "tracking number you sent doesn't work. I need this fixed today.'"
)
BATCH_EXAMPLE_QUESTIONS = "\n".join([
    "choice | Which team should own this ticket? | billing, shipping, support, retention",
    "score | How urgent is this ticket, 0=low..3=critical? | 0, 1, 2, 3",
    "noul | Should this be escalated to a manager? | ",
    "choice | What does the customer want? | refund, replacement, apology",
])

# ---------------------------------------------------------------------------
# Blocks layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Typical") as demo:
    gr.Markdown(HEADER)
    model_dd = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
    status_md = gr.Markdown("Not loaded yet -- pick a model or press Run.")
    model_dd.change(load_status, model_dd, status_md)

    with gr.Tab("Playground"):
        with gr.Row():
            with gr.Column():
                state_tb = gr.Textbox(lines=6, label="State")
                question_tb = gr.Textbox(lines=2, label="Question / rubric")
                primitive_radio = gr.Radio(["Choice", "Score", "Noul"], value="Choice", label="Primitive")
                labels_tb = gr.Textbox(label="Labels (comma-separated; ordered levels for Score)",
                                       placeholder="refund, replacement, repair")
                primitive_radio.change(toggle_labels, primitive_radio, labels_tb)
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                bar = gr.BarPlot(x="probability", y="label", label="Probability distribution", height=300)
                summary_md = gr.Markdown()
        run_btn.click(run_playground, [model_dd, state_tb, question_tb, primitive_radio, labels_tb],
                      [bar, summary_md])
        gr.Examples(EXAMPLES, inputs=[state_tb, question_tb, primitive_radio, labels_tb])

    with gr.Tab("Batch"):
        gr.Markdown("One state, several questions -- all scored against a **single KV-encode** of "
                    "the state (`native_kv_decide`), showing what the cached state buys you.")
        batch_state_tb = gr.Textbox(lines=4, label="State", value=BATCH_EXAMPLE_STATE)
        batch_questions_tb = gr.Textbox(lines=6, label="Questions (one per line: type | question | labels)",
                                        value=BATCH_EXAMPLE_QUESTIONS)
        batch_run_btn = gr.Button("Run batch", variant="primary")
        batch_table = gr.Dataframe(headers=["type", "question", "argmax", "p", "p_null"])
        batch_summary_md = gr.Markdown()
        batch_run_btn.click(run_batch, [model_dd, batch_state_tb, batch_questions_tb],
                            [batch_table, batch_summary_md])

    with gr.Tab("Results"):
        gr.Markdown(RESULTS_MD)

if __name__ == "__main__":
    demo.launch()
