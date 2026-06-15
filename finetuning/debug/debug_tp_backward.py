# coding=utf-8
"""Minimal test: trace where NaN first appears in backward when text_projection is trainable."""
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


def make_nan_hook(name):
    """Hook that prints the FIRST occurrence of NaN in backward."""
    first = [True]
    def hook(module, grad_input, grad_output):
        for i, g in enumerate(grad_output):
            if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                if first[0]:
                    print(f"  FIRST NaN/Inf at: {name} grad_output[{i}] shape={g.shape}")
                    first[0] = False
        for i, g in enumerate(grad_input):
            if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                if first[0]:
                    print(f"  FIRST NaN/Inf at: {name} grad_input[{i}] shape={g.shape}")
                    first[0] = False
    return hook


def test(train_text_projection):
    label = "WITH" if train_text_projection else "WITHOUT"
    print(f"\n{'='*60}")
    print(f"Testing {label} trainable text_projection")
    print(f"{'='*60}")

    model = Qwen3TTSModel.from_pretrained(
        "e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    config = AutoConfig.from_pretrained("e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

    for p in model.model.parameters():
        p.requires_grad = False

    # Apply LoRA
    peft_config = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=False,
    )
    model.model.talker.model = inject_adapter_in_model(peft_config, model.model.talker.model)

    # Only layers 18+ trainable
    for n, p in model.model.named_parameters():
        if "lora_" in n:
            marker = "talker.model.layers."
            if marker in n:
                tail = n.split(marker, 1)[1]
                layer_idx = int(tail.split(".", 1)[0])
                p.requires_grad = layer_idx >= 18

    # text_projection
    for p in model.model.talker.text_projection.parameters():
        p.requires_grad = train_text_projection

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Register NaN hooks on ALL modules
    for nm, mod in model.model.named_modules():
        if isinstance(mod, (nn.Linear, nn.RMSNorm)) or "RMSNorm" in type(mod).__name__:
            mod.register_full_backward_hook(make_nan_hook(nm))

    # Load one sample
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

    # Forward
    input_text_embedding = model.model.talker.text_projection(
        model.model.talker.model.text_embedding(input_text_ids)
    ) * text_embedding_mask
    input_codec_embedding = model.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_embeddings = input_text_embedding + input_codec_embedding

    # Skip aux codec for simplicity
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
    print(f"main_loss={outputs.loss.item():.4f}  sub_loss={sub_talker_loss.item():.4f}  "
          f"total={loss.item():.4f}  NaN={torch.isnan(loss).item()}")

    if torch.isnan(loss):
        print("Loss is NaN, skipping backward")
        return

    print("Running backward...")
    loss.backward()

    # Check which params have NaN
    nan_params = []
    for n, p in model.model.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            nan_params.append(n)
    if nan_params:
        print(f"NaN/Inf gradients ({len(nan_params)} params):")
        for n in nan_params[:10]:
            print(f"  {n}")
        if len(nan_params) > 10:
            print(f"  ... and {len(nan_params) - 10} more")
    else:
        # Print grad norms for trainable params
        print("All gradients clean. Trainable grad norms:")
        for n, p in model.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                print(f"  {n}: grad_norm={p.grad.norm().item():.6f}")


if __name__ == "__main__":
    test(train_text_projection=False)
    test(train_text_projection=True)
