"""REVIEW.md #5 item 1: untrained nearest-label cosine accuracy on held-out label
spaces, no training, $0. Three candidate encoders x {raw, "intent: X"} label wording:

  1. layer-12 mean-pooled features -- what the trained model actually reads
     (Backbone(lora_layers=8, lora_r=0, tap_layer=20) -> split_layer 12).
  2. layer-20 mean-pooled features (tap_layer=20, lora_layers=0 -> split_layer 20).
  3. Qwen3-Embedding-0.6B, last-token pooling per its model card (instruct prefix on
     the state/query side, raw text on the candidate/document side).

Both (1) and (2) share one Qwen3-1.7B-Base forward pass (pooled_multi taps two
hidden_states indices per batch instead of loading the backbone twice).

Reports among-K accuracy (null rows dropped, argmax over each row's own candidate
list -- same definition as metrics.summarize's acc_k) next to the trained model's
acc_k from runs/joint_v1 and runs/main_v3, for the same eval sets.

uv run scripts/cosine_probe.py [--limit 1000] [--device auto]
uv run scripts/cosine_probe.py --selftest
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encode import Backbone, pick_device

EVAL_SETS = ["banking77_k", "banking77_test", "trec_coarse", "trec_fine", "ng20_test", "clinc_heldout"]
REF_RUNS = {"joint_v1": "runs/joint_v1/results.json", "main_v3": "runs/main_v3/results.json"}
LABEL_STYLES = {"raw": "{}", "rendered": "intent: {}"}
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_INSTRUCT = "Instruct: Given a piece of text, retrieve the label that best describes it\nQuery: {}"


def load_capped(name, limit, data_dir="data"):
    rows = [json.loads(l) for l in open(f"{data_dir}/eval/{name}.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("label") not in (None, -1)]  # among-K only, ignore null
    return rows[:limit]


@torch.inference_mode()
def pooled_multi(backbone, texts, layers, max_len, batch_size=64):
    """-> {layer: [len(texts), d] float32}, mean-pooled over real tokens, one forward
    pass per batch shared across all `layers` (mirrors Backbone.frozen_features but
    taps several hidden_states indices instead of only self.split_layer)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = {L: [None] * len(texts) for L in layers}
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        input_ids, attention_mask = backbone.tokenize([texts[i] for i in idx], max_len)
        lengths = attention_mask.sum(1).tolist()  # CPU: avoid per-item MPS sync
        input_ids, attention_mask = input_ids.to(backbone.device), attention_mask.to(backbone.device)
        hs = backbone.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True).hidden_states
        for L in layers:
            h = backbone.model.norm(hs[L])
            for j, i in enumerate(idx):
                out[L][i] = h[j, 1:lengths[j]].float().mean(0).cpu()
    return {L: torch.stack(v) for L, v in out.items()}


@torch.inference_mode()
def embed_last_token(model, tok, texts, device, instruct_fmt=None, max_len=256, batch_size=64):
    """Last-token pooling, L2-normalised, per Qwen3-Embedding's model card. instruct_fmt,
    when given, wraps each text (query side); candidate/document side passes None."""
    texts = [instruct_fmt.format(t) if instruct_fmt else t for t in texts]
    texts = [t + tok.eos_token for t in texts]
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, truncation=True, max_length=max_len, padding=True,
                   add_special_tokens=False, return_tensors="pt")
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
        seq_lens = am.sum(1) - 1
        pooled = h[torch.arange(h.shape[0], device=device), seq_lens]
        out.append(F.normalize(pooled.float(), p=2, dim=-1).cpu())
    return torch.cat(out)


def among_k_acc(rows, state_of, label_of):
    """state_of[i] / label_of[text] -> unit-normalised vector. Argmax cosine per row's
    own candidate list, same acc_k definition as metrics.summarize."""
    correct = 0
    for i, row in enumerate(rows):
        cvecs = torch.stack([label_of[c] for c in row["candidates"]])
        sims = F.cosine_similarity(state_of[i].unsqueeze(0), cvecs, dim=-1)
        correct += int(sims.argmax().item() == row["label"])
    return correct / len(rows)


def load_ref_acc_k(path, set_name):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return "n/a"
    v = d.get("eval", {}).get(set_name, {}).get("scaled", {}).get("acc_k", "n/a")
    return v


def demo():
    """Self-check: pooled_multi's two tap layers actually differ (not the same tensor
    twice), and cosine argmax picks the exact-match candidate. No 1.7B download needed."""
    device = pick_device("auto")
    bb = Backbone(name="Qwen/Qwen3-0.6B-Base", lora_layers=0, lora_r=0, device=device)
    n_layers = bb.model.config.num_hidden_layers
    feats = pooled_multi(bb, ["a cat sat on a mat", "quarterly revenue guidance"],
                          layers=[n_layers // 2, n_layers], max_len=32, batch_size=8)
    lo, hi = feats[n_layers // 2], feats[n_layers]
    assert lo.shape == hi.shape == (2, bb.d)
    assert not torch.allclose(lo, hi), "layer-12/layer-20 taps returned identical features"
    print(f"PASS: pooled_multi taps layer {n_layers // 2} and {n_layers} distinctly, shape {tuple(lo.shape)}")

    sv = F.normalize(hi[0:1], dim=-1)[0]
    label_of = {"cat": F.normalize(hi[0:1], dim=-1)[0], "revenue": F.normalize(hi[1:2], dim=-1)[0]}
    row = {"candidates": ["cat", "revenue"], "label": 0}
    acc = among_k_acc([row], [sv], label_of)
    assert acc == 1.0
    print("PASS: among_k_acc picks the identical-vector candidate")
    print("cosine_probe.py selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backbone", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return demo()

    t0 = time.time()
    device = pick_device(args.device)

    print(f"[1/4] loading eval sets (cap={args.limit}, null rows dropped) ...")
    eval_rows = {name: load_capped(name, args.limit) for name in EVAL_SETS}
    for name, rows in eval_rows.items():
        print(f"  {name}: {len(rows)} among-K rows")

    print(f"[2/4] {args.backbone} tap_layer=20, layers {{12, 20}} ...")
    bb = Backbone(name=args.backbone, lora_layers=0, lora_r=0, tap_layer=20, device=device)
    layers = [12, 20]

    all_states = sorted({r["state"] for rows in eval_rows.values() for r in rows})
    all_labels = sorted({c for rows in eval_rows.values() for r in rows for c in r["candidates"]})
    print(f"  encoding {len(all_states)} unique states ...")
    state_feats = pooled_multi(bb, all_states, layers, max_len=256)
    state_vecs = {L: {t: F.normalize(v, dim=-1) for t, v in zip(all_states, state_feats[L])} for L in layers}

    label_vecs = {L: {} for L in layers}  # (layer, style) -> {label_text: vec}
    for style, fmt in LABEL_STYLES.items():
        rendered = [fmt.format(c) for c in all_labels]
        print(f"  encoding {len(all_labels)} unique labels, style={style} ...")
        feats = pooled_multi(bb, rendered, layers, max_len=24)
        for L in layers:
            label_vecs[L][style] = {c: F.normalize(v, dim=-1) for c, v in zip(all_labels, feats[L])}
    del bb

    print(f"[3/4] {EMBED_MODEL} last-token pooling ...")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="right")
    emb_bb = Backbone(name=EMBED_MODEL, lora_layers=0, lora_r=0, device=device)  # reuse loader only
    emb_model = emb_bb.model
    print(f"  encoding {len(all_states)} unique states (instructed) ...")
    ev_states = embed_last_token(emb_model, tok, all_states, device, instruct_fmt=EMBED_INSTRUCT)
    emb_state_vecs = dict(zip(all_states, ev_states))
    emb_label_vecs = {}
    for style, fmt in LABEL_STYLES.items():
        rendered = [fmt.format(c) for c in all_labels]
        print(f"  encoding {len(all_labels)} unique labels, style={style} ...")
        ev_labels = embed_last_token(emb_model, tok, rendered, device, instruct_fmt=None, max_len=24)
        emb_label_vecs[style] = dict(zip(all_labels, ev_labels))
    del emb_bb, emb_model

    print("[4/4] scoring ...\n")
    cols = [f"layer12_{s}" for s in LABEL_STYLES] + [f"layer20_{s}" for s in LABEL_STYLES] + \
           [f"qwen3emb_{s}" for s in LABEL_STYLES]
    header = f"{'set':<16}" + "".join(f"{c:>20}" for c in cols) + "".join(f"{r:>14}" for r in REF_RUNS)
    print(header)
    table = {}
    for name, rows in eval_rows.items():
        vals = []
        for L in layers:
            for style in LABEL_STYLES:
                sv = [state_vecs[L][r["state"]] for r in rows]
                acc = among_k_acc(rows, sv, label_vecs[L][style])
                vals.append(acc)
        for style in LABEL_STYLES:
            sv = [emb_state_vecs[r["state"]] for r in rows]
            acc = among_k_acc(rows, sv, emb_label_vecs[style])
            vals.append(acc)
        refs = [load_ref_acc_k(REF_RUNS[r], name) for r in REF_RUNS]
        row_str = f"{name:<16}" + "".join(f"{v:>20.3f}" for v in vals) + \
                  "".join(f"{r:>14.3f}" if isinstance(r, float) else f"{r:>14}" for r in refs)
        print(row_str)
        table[name] = dict(zip(cols, vals), **{f"trained_{k}": v for k, v in zip(REF_RUNS, refs)})

    print(f"\ndone in {time.time() - t0:.1f}s")
    return table


if __name__ == "__main__":
    main()
