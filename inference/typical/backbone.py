"""Frozen backbone + LoRA adapter -- trimmed port of the training repo's encode.py, for
inference only. Only what native_kv_decide touches: tokenizer, .model (the raw HF causal
trunk, tap-truncated), LoRA weight loading. No FeatureCache/EmbedEncoder/forward() --
the native serving path reads hidden states straight off Backbone.model, never through a
Backbone.forward() (that method, and the frozen-feature/h_top machinery it supports, only
exists in the training repo's energy-head path). extra_tap is also dropped: both shipped
checkpoints (typical-small, typical-small-preview) train with extra_tap=0.
"""
import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Qwen3: q/k/v/o_proj (self_attn) + gate/up/down_proj (mlp). Qwen3.5 hybrid stack adds
# in_proj_{qkv,z,b,a}/out_proj for its Gated-DeltaNet linear-attention layers (linear_attn,
# no self_attn) -- listed here too since hasattr() gates each name per parent module, so this
# tuple is a superset that's a no-op for any module lacking a given name (Qwen3 untouched).
_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
_printed_device = False


def pick_device(device: str = "auto") -> str:
    global _printed_device
    if device != "auto":
        resolved = device
    elif torch.cuda.is_available():
        resolved = "cuda"
    elif torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
    if not _printed_device:
        print(f"typical: device={resolved}")
        _printed_device = True
    return resolved


class LoRALinear(nn.Module):
    """Frozen base nn.Linear (bf16) + a low-rank fp32 adapter on top. Numerics verbatim
    from encode.py's LoRALinear (scale = alpha / r); dropout is a no-op once .eval()'d."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, x):
        out = self.base(x)
        lora = self.drop(x) @ self.A.to(x.dtype).T @ self.B.to(x.dtype).T * self.scale
        return out + lora


class Backbone(nn.Module):
    """Frozen causal-LM trunk (Qwen3-family), truncated to tap_layer, with LoRA on its top
    lora_layers blocks. Base weights come fresh from `name`; the fine-tuned delta is only
    ever the LoRA A/B tensors, loaded via load_lora_state_dict from the checkpoint."""

    def __init__(self, name: str, lora_layers: int = 8, lora_r: int = 16, lora_alpha: float = 32,
                 lora_dropout: float = 0.05, device: str = "auto", tap_layer: int = 0):
        super().__init__()
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
        # Qwen3.5 family: AutoModel resolves to the VL wrapper (.visual + .language_model) even for
        # a text-only checkpoint -- see encode.py's Backbone for the long form. Take the text trunk.
        if hasattr(model, "language_model") and hasattr(model, "visual"):
            model = model.language_model
        self.model = model
        self.model.requires_grad_(False)
        self.d = self.model.config.hidden_size
        n_layers = self.model.config.num_hidden_layers
        # tap_layer: drop every layer above T (semantics live mid-depth, not at the final,
        # next-token-shaped layer of a base LM) -- see encode.py's Backbone for the long form.
        if 0 < tap_layer < n_layers:
            self.model.layers = self.model.layers[:tap_layer]
            self.model.config.num_hidden_layers = tap_layer
            n_layers = tap_layer
        self.info = {"n_layers": n_layers,
                     "layer_types": list(getattr(self.model.config, "layer_types", None)
                                          or ["full_attention"] * n_layers)[:n_layers]}
        if lora_r > 0:
            for layer in self.model.layers[-lora_layers:]:
                # Qwen3.5 linear_attention layers have linear_attn (Gated DeltaNet) instead of
                # self_attn; mlp is present on every layer either way -- see encode.py's Backbone.
                parents = [p for p in (getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None),
                                        getattr(layer, "mlp", None)) if p is not None]
                for parent in parents:
                    for mod_name in _LORA_MODULES:
                        if hasattr(parent, mod_name):
                            setattr(parent, mod_name,
                                    LoRALinear(getattr(parent, mod_name), lora_r, lora_alpha, lora_dropout))
        self.to(self.device)
        self.eval()

    def load_lora_state_dict(self, sd: dict):
        own = dict(self.named_parameters())
        for k, v in sd.items():
            own[k].data.copy_(v.to(own[k].device, own[k].dtype))
