# coding=utf-8
"""
LoRA fine-tuning for Qwen3-TTS VoiceDesign model.

Differs from sft_12hz.py (Base→CustomVoice) in:
  - Loads VoiceDesign model (no speaker_encoder)
  - Prepends instruction embeddings at sequence start
  - No speaker_embedding insertion (voice is controlled by instruction)
  - Uses LoRA for memory efficiency (fits 16GB VRAM)
  - Saves LoRA adapters instead of full model checkpoints
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


class VoiceDesignDataset(Dataset):
    """Dataset for VoiceDesign fine-tuning: each sample = (text, instruct, audio_codes)."""

    def __init__(self, data_list, processor, config):
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _build_assistant_text(self, text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct_text(self, instruct):
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize(self, text):
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        instruct = item.get("instruct", "")
        audio_codes = item["audio_codes"]

        text_str = self._build_assistant_text(text)
        text_ids = self._tokenize(text_str)

        instruct_str = self._build_instruct_text(instruct) if instruct else ""
        instruct_ids = self._tokenize(instruct_str) if instruct_str else torch.zeros((1, 0), dtype=torch.long)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        return {
            "text_ids": text_ids[:, :-5],
            "instruct_ids": instruct_ids,
            "audio_codes": audio_codes,
        }

    def collate_fn(self, batch):
        """
        Build input tensors for VoiceDesign training.

        Sequence layout (no speaker embedding):
          [instruct] [role(3)] [prefill(5)] [BOS] [text...] [EOS] [pad] [cBOS] [codec...] [cEOS]

        The 5 codec prefill positions: nothink, think_bos, think_eos, 0, pad
        """
        cfg = self.config
        tcfg = cfg.talker_config

        instruct_lens = [b["instruct_ids"].shape[1] for b in batch]
        max_instruct = max(instruct_lens)

        lengths = [
            il + b["text_ids"].shape[1] + b["audio_codes"].shape[0]
            for il, b in zip(instruct_lens, batch)
        ]
        max_total = max(lengths) + 10

        B, T = len(batch), max_total
        input_text_ids = torch.zeros((B, T), dtype=torch.long)
        input_codec_ids = torch.zeros((B, T), dtype=torch.long)
        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        codec_0_labels = torch.full((B, T), -100, dtype=torch.long)
        codec_ids_full = torch.zeros((B, T, 16), dtype=torch.long)

        for i, data in enumerate(batch):
            instruct_ids = data["instruct_ids"][0]
            text_ids = data["text_ids"][0]
            audio_codes = data["audio_codes"]
            codec0 = audio_codes[:, 0]
            I = instruct_ids.shape[0]
            Tlen = text_ids.shape[0]
            C = codec0.shape[0]

            # --- Text channel ---
            # Instruction
            if I > 0:
                input_text_ids[i, :I] = instruct_ids
            # Role tokens: <|im_start|>assistant\n  (first 3 tokens of assistant text)
            input_text_ids[i, I:I + 3] = text_ids[:3]
            # Codec prefill (5 positions): pad tokens on text side
            input_text_ids[i, I + 3:I + 8] = cfg.tts_pad_token_id
            # BOS
            input_text_ids[i, I + 8] = cfg.tts_bos_token_id
            # Text body
            input_text_ids[i, I + 9:I + 9 + Tlen - 3] = text_ids[3:]
            # EOS
            input_text_ids[i, I + 9 + Tlen - 3] = cfg.tts_eos_token_id
            # Padding during codec
            input_text_ids[i, I + 9 + Tlen - 2:I + 9 + Tlen + C] = cfg.tts_pad_token_id

            text_embedding_mask[i, I:I + 9 + Tlen + C] = True

            # --- Codec channel ---
            # Prefill: nothink, think_bos, think_eos, 0, pad
            input_codec_ids[i, I + 3] = tcfg.codec_nothink_id
            input_codec_ids[i, I + 4] = tcfg.codec_think_bos_id
            input_codec_ids[i, I + 5] = tcfg.codec_think_eos_id
            input_codec_ids[i, I + 6] = 0
            input_codec_ids[i, I + 7] = tcfg.codec_pad_id
            # Padding during text
            input_codec_ids[i, I + 8:I + 9 + Tlen - 2] = tcfg.codec_pad_id
            # BOS for codec
            input_codec_ids[i, I + 9 + Tlen - 2] = tcfg.codec_bos_id
            # Audio codec tokens
            input_codec_ids[i, I + 9 + Tlen - 1:I + 9 + Tlen - 1 + C] = codec0
            # EOS
            input_codec_ids[i, I + 9 + Tlen - 1 + C] = tcfg.codec_eos_token_id

            codec_embedding_mask[i, I + 3:I + 9 + Tlen + C + 1] = True
            codec_embedding_mask[i, I + 6] = False  # skip speaker slot (no speaker for VoiceDesign)

            codec_mask[i, I + 9 + Tlen - 1:I + 9 + Tlen - 1 + C] = True

            codec_0_labels[i, I + 9 + Tlen - 1:I + 9 + Tlen - 1 + C] = codec0
            codec_0_labels[i, I + 9 + Tlen - 1 + C] = tcfg.codec_eos_token_id

            codec_ids_full[i, I + 9 + Tlen - 1:I + 9 + Tlen - 1 + C, :] = audio_codes

            attention_mask[i, :I + 9 + Tlen + C + 1] = True

        return {
            "input_text_ids": input_text_ids,
            "input_codec_ids": input_codec_ids,
            "codec_ids": codec_ids_full,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "attention_mask": attention_mask,
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
        }


def apply_lora(model, r=8, alpha=16, dropout=0.05, layer_start=18, train_text_projection=False):
    """Apply LoRA to main transformer only (NOT code_predictor — that causes NaN)."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=False,
    )

    # Only apply LoRA to the main talker transformer, NOT to code_predictor.
    # The code_predictor is a separate small transformer that predicts auxiliary
    # codebooks from the main model's hidden states. Applying LoRA to it
    # destabilizes the sub_talker loss.
    model.talker.model = inject_adapter_in_model(peft_config, model.talker.model)

    for p in model.talker.text_projection.parameters():
        p.requires_grad = train_text_projection

    # Mark only selected LoRA layers as trainable. Earlier layers have unstable
    # backward in the released VoiceDesign checkpoint on this setup.
    for n, p in model.named_parameters():
        if "lora_" in n:
            layer_idx = None
            marker = "talker.model.layers."
            if marker in n:
                tail = n.split(marker, 1)[1]
                layer_idx = int(tail.split(".", 1)[0])
            p.requires_grad = layer_idx is not None and layer_idx >= layer_start

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output_model_path", type=str, default="output_vd")
    parser.add_argument("--train_jsonl", type=str, default="train_with_codes.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layer_start", type=int, default=18)
    parser.add_argument("--train_text_projection", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--skip_sub_loss", action="store_true")
    parser.add_argument("--skip_aux_codec", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"])
    parser.add_argument("--model_dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )

    MODEL_PATH = args.init_model_path
    if accelerator.is_main_process:
        print(f"Loading VoiceDesign model from {MODEL_PATH} ...")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32 if args.model_dtype == "float32" else torch.bfloat16,
        attn_implementation="eager",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Freeze all
    for p in qwen3tts.model.parameters():
        p.requires_grad = False

    # Apply LoRA (main transformer only) + unfreeze text_projection
    qwen3tts.model = apply_lora(
        qwen3tts.model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        layer_start=args.lora_layer_start,
        train_text_projection=args.train_text_projection,
    )

    # Check LoRA parameter dtypes
    if accelerator.is_main_process:
        lora_dtypes = set()
        for n, p in qwen3tts.model.named_parameters():
            if "lora_" in n:
                lora_dtypes.add(str(p.dtype))
        base_dtype = str(next(qwen3tts.model.parameters()).dtype)
        print(f"Base model dtype: {base_dtype}")
        print(f"LoRA dtypes: {lora_dtypes}")

    # Gradient checkpointing disabled — can cause NaN in backward with bf16 + LoRA
    # The model fits 16GB without it (batch_size=1, bf16, LoRA only)

    # Data
    train_data = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8")]
    dataset = VoiceDesignDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )
    if args.smoke_test:
        batch = next(iter(train_dataloader))
        if accelerator.is_main_process:
            print("Smoke test OK:")
            for k, v in batch.items():
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        return

    trainable_params = [p for p in qwen3tts.model.parameters() if p.requires_grad]
    # Higher epsilon for bf16 stability (default 1e-8 can underflow)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01, eps=1e-6)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    # Verify trainable params survived accelerator.prepare
    trainable_after = [n for n, p in model.named_parameters() if p.requires_grad]
    if accelerator.is_main_process:
        print(f"Trainable params after prepare: {len(trainable_after)}")
        if len(trainable_after) == 0:
            print("ERROR: No trainable parameters after accelerator.prepare!")
            return

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_text_ids = batch["input_text_ids"]
                input_codec_ids = batch["input_codec_ids"]
                codec_ids = batch["codec_ids"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                # Apply text_projection — matches inference path (generate_voice_design)
                input_text_embedding = model.talker.text_projection(
                    model.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_embeddings = input_text_embedding + input_codec_embedding

                if not args.skip_aux_codec:
                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                # Clip Inf gradients at transformer input to protect text_projection.
                # Frozen pretrained attention projections (sv_min ~2e-5) produce
                # Inf gradients that propagate backward. They don't affect LoRA
                # directly, but would flow into text_projection's backward graph.
                # torch.nan_to_num caps Inf→1e4, NaN→0 at the boundary.
                def _clip_inf_hook(grad):
                    return torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)

                # Clip at the transformer input (for text_projection gradient)
                if input_embeddings.requires_grad:
                    input_embeddings.register_hook(_clip_inf_hook)

                # FIX: Do NOT double-shift. The model's loss_function internally
                # shifts logits/labels by 1. Passing pre-shifted data causes
                # the loss to be computed on misaligned token pairs.
                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask]
                talker_codec_ids = codec_ids[codec_mask]

                if args.skip_sub_loss:
                    sub_talker_loss = torch.tensor(0.0, device=input_text_ids.device)
                else:
                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                main_loss = outputs.loss
                sub_loss = sub_talker_loss

                if torch.isnan(main_loss) or torch.isinf(main_loss):
                    accelerator.print(f"WARNING: main_loss NaN/Inf at step {step}, skipping")
                    optimizer.zero_grad()
                    continue

                if torch.isnan(sub_loss) or torch.isinf(sub_loss):
                    accelerator.print(f"WARNING: sub_loss NaN/Inf at step {step}, using main_loss only")
                    loss = main_loss
                else:
                    loss = main_loss + 0.3 * sub_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if grad_norm is not None:
                        gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                        accelerator.print(f"  step {step} grad_norm: {gn:.4f}")
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            accelerator.print(f"  CRITICAL: grad_norm is NaN/Inf!")
                            for n, p in model.named_parameters():
                                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                    accelerator.print(f"    NaN/Inf grad in: {n}  shape={p.shape}")

                optimizer.step()
                optimizer.zero_grad()

            if step % 5 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | main: {main_loss.item():.4f} | sub: {sub_loss.item():.4f}"
                )
            if args.max_steps is not None and step + 1 >= args.max_steps:
                break

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            state = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()
                     if ("lora_" in k or "text_projection" in k)}
            save_file(state, os.path.join(output_dir, "lora_adapter.safetensors"))
            shutil.copy(os.path.join(MODEL_PATH, "config.json"),
                       os.path.join(output_dir, "config.json"))
            accelerator.print(f"Saved checkpoint to {output_dir}")
        if args.max_steps is not None:
            break


if __name__ == "__main__":
    train()
