# coding=utf-8
"""Diagnose RMSNorm weights and attention projection conditioning in the VoiceDesign model."""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model.safetensors"

state = load_file(MODEL_PATH)

# Collect RMSNorm weights
rms_weights = {}
for k, v in state.items():
    if "norm.weight" in k or "q_norm" in k or "k_norm" in k:
        rms_weights[k] = v

print("=== RMSNorm Weight Statistics ===")
for k, v in sorted(rms_weights.items(), key=lambda x: x[0]):
    w = v.float()
    print(f"  {k}: shape={list(v.shape)} mean={w.mean().item():.4f} std={w.std().item():.4f} "
          f"min={w.min().item():.4f} max={w.max().item():.4f}")

# Check attention projection condition numbers for problematic layers
print("\n=== Attention Projection Condition Numbers (first 10 singular values) ===")
for layer_idx in [0, 3, 15, 16, 26, 27]:
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        key = f"talker.model.layers.{layer_idx}.self_attn.{proj}.weight"
        if key in state:
            w = state[key].float()
            # Use randomized SVD for large matrices
            with torch.no_grad():
                _, S, _ = torch.linalg.svd(w, full_matrices=False)
                cond = (S[0] / S[-1]).item()
                print(f"  layer {layer_idx} {proj}: shape={list(w.shape)} "
                      f"sv_max={S[0].item():.2f} sv_min={S[-1].item():.6f} cond={cond:.1f}")

# Check hidden state norms through one forward pass
print("\n=== Forward Hidden State Norms ===")
from transformers import AutoConfig
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    torch_dtype=torch.float32,
    attn_implementation="eager",
    device_map="cuda:0",
)
config = AutoConfig.from_pretrained("e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

# Hook to capture hidden states
hs_norms = {}
def make_hs_hook(layer_idx):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        hs_norms[layer_idx] = out.norm().item()
    return hook

for i, layer in enumerate(model.model.talker.model.layers):
    layer.register_forward_hook(make_hs_hook(i))

# Dummy forward pass
B, T = 1, 128
dummy_embeds = torch.randn(B, T, 2048, device="cuda:0", dtype=torch.float32)
dummy_mask = torch.ones(B, T, device="cuda:0", dtype=torch.long)
with torch.no_grad():
    model.model.talker.model(inputs_embeds=dummy_embeds, attention_mask=dummy_mask)

for i in sorted(hs_norms.keys()):
    print(f"  layer {i}: hidden_state norm = {hs_norms[i]:.2f}")

# Test: what happens if we reset RMSNorm weights to 1.0?
print("\n=== RMSNorm weights: reset-to-1.0 simulation ===")
for k, v in sorted(rms_weights.items(), key=lambda x: x[0]):
    w = v.float()
    if w.mean() > 2.0 or w.mean() < 0.5:
        print(f"  {k}: mean={w.mean().item():.4f} -> WOULD RESET")
