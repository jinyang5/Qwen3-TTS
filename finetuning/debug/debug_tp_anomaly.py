# coding=utf-8
"""Use torch.autograd.detect_anomaly to find exact NaN-producing op."""
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

# Enable anomaly detection BEFORE any operations
torch.autograd.set_detect_anomaly(True)

# Load model (same setup as training)
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
    p.requires_grad = True

# Load sample
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

print("Running backward with anomaly detection...")
try:
    loss.backward()
    print("Backward completed without error")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
