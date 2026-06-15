# coding=utf-8
"""Compare gradient norms with and without text_projection to find divergence point."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig
from peft import LoraConfig, inject_adapter_in_model


def run_test(train_tp, label):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    model = Qwen3TTSModel.from_pretrained(
        "e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    config = AutoConfig.from_pretrained("e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

    for p in model.model.parameters():
        p.requires_grad = False

    peft_config = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=False,
    )
    model.model.talker.model = inject_adapter_in_model(peft_config, model.model.talker.model)

    for n, p in model.model.named_parameters():
        if "lora_" in n:
            marker = "talker.model.layers."
            if marker in n:
                tail = n.split(marker, 1)[1]
                layer_idx = int(tail.split(".", 1)[0])
                p.requires_grad = layer_idx >= 18

    for p in model.model.talker.text_projection.parameters():
        p.requires_grad = train_tp

    # Track gradient norms at critical points
    norms = {}

    def make_norm_hook(name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_output):
                if g is not None:
                    norms[f"{name}.grad_out"] = g.norm().item()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        norms[f"{name}.grad_out_NAN"] = True
            for i, g in enumerate(grad_input):
                if g is not None:
                    norms[f"{name}.grad_in"] = g.norm().item()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        norms[f"{name}.grad_in_NAN"] = True
        return hook

    # Hook only layers 16-19 (where the NaN first appears)
    for i in [16, 17, 18, 19]:
        for sub in ["input_layernorm", "post_attention_layernorm"]:
            name = f"layer.{i}.{sub}"
            module = model.model.talker.model.layers[i]
            if sub == "input_layernorm":
                getattr(module, "input_layernorm").register_full_backward_hook(make_norm_hook(name))
            else:
                getattr(module, "post_attention_layernorm").register_full_backward_hook(make_norm_hook(name))
        # Also hook MLP down_proj
        for sub in ["down_proj", "gate_proj", "up_proj"]:
            name = f"layer.{i}.mlp.{sub}"
            getattr(module.mlp, sub).register_full_backward_hook(make_norm_hook(name))
        # And attention projections
        for sub in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            name = f"layer.{i}.attn.{sub}"
            getattr(module.self_attn, sub).register_full_backward_hook(make_norm_hook(name))

    # Load data
    data = [json.loads(line) for line in open("train_with_codes.jsonl", encoding="utf-8")]
    from sft_voice_design import VoiceDesignDataset
    dataset = VoiceDesignDataset(data, model.processor, config)
    batch = dataset.collate_fn([dataset[0]])

    input_text_ids = batch["input_text_ids"]
    input_codec_ids = batch["input_codec_ids"]
    codec_ids = batch["codec_ids"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    input_text_embedding = model.model.talker.text_projection(
        model.model.talker.model.text_embedding(input_text_ids)
    ) * text_embedding_mask
    input_codec_embedding = model.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_embeddings = input_text_embedding + input_codec_embedding

    outputs = model.model.talker(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=codec_0_labels,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask]
    talker_codec_ids = codec_ids[codec_mask]

    sub_talker_logits, sub_talker_loss = model.model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )

    loss = outputs.loss + 0.3 * sub_talker_loss
    print(f"loss={loss.item():.4f}")
    loss.backward()

    # Print norms sorted
    nan_keys = [k for k, v in norms.items() if "NAN" in k]
    if nan_keys:
        print(f"\nNaN/Inf detected in: {nan_keys}")

    norm_items = [(k, v) for k, v in norms.items() if "NAN" not in k]
    norm_items.sort()
    print(f"\nGradient norms:")
    for k, v in norm_items:
        print(f"  {k}: {v:.2f}")

    return norms


norms_without = run_test(train_tp=False, label="WITHOUT text_projection")
norms_with = run_test(train_tp=True, label="WITH text_projection")

# Compare
print(f"\n{'='*60}")
print("COMPARISON (ratio: with/without)")
print(f"{'='*60}")
for k in sorted(norms_without.keys()):
    if "NAN" in k:
        continue
    v_wo = norms_without.get(k, 0)
    v_w = norms_with.get(k, 0)
    if v_wo > 0:
        ratio = v_w / v_wo
        marker = " *** DIVERGENT" if ratio > 10 or ratio < 0.1 else ""
        print(f"  {k}: {v_wo:.2f} -> {v_w:.2f}  (ratio={ratio:.2f}){marker}")
    else:
        print(f"  {k}: {v_wo:.2f} -> {v_w:.2f}")
