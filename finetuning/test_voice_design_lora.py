# coding=utf-8
"""Run inference with a VoiceDesign LoRA checkpoint.

The finetuning script saves LoRA weights separately as `lora_adapter.safetensors`,
so testing requires loading the base VoiceDesign model, injecting the same LoRA
modules, then loading the adapter weights.
"""
import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel


def inject_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, inject_adapter_in_model

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=False,
    )
    model.talker.model = inject_adapter_in_model(peft_config, model.talker.model)
    return model


def load_lora_checkpoint(tts, checkpoint_dir: str, r: int, alpha: int, dropout: float):
    adapter_path = os.path.join(checkpoint_dir, "lora_adapter.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Missing adapter file: {adapter_path}")

    tts.model = inject_lora(tts.model, r=r, alpha=alpha, dropout=dropout)
    state = load_file(adapter_path)
    missing, unexpected = tts.model.load_state_dict(state, strict=False)
    print(f"Loaded adapter: {adapter_path}")
    print(f"  adapter tensors: {len(state)}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if unexpected:
        print("  unexpected examples:", unexpected[:5])
    return tts


def generate_one(tts, text: str, instruct: str, language: str, output_wav: str, max_new_tokens: int):
    wavs, sr = tts.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        max_new_tokens=max_new_tokens,
    )
    sf.write(output_wav, wavs[0], sr)
    print(f"Saved: {output_wav} ({sr} Hz)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--checkpoint_dir", type=str, default="output_vd_1000/checkpoint-epoch-4")
    parser.add_argument("--output_dir", type=str, default="test_outputs_vd_1000")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--compare_base", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32

    samples = [
        {
            "name": "zh_warm",
            "language": "Chinese",
            "text": "今天的安排已经确认好了，我们可以按计划开始准备。",
            "instruct": "用温柔、自然的年轻女声说，语速中等，语气亲切。",
        },
        {
            "name": "zh_firm",
            "language": "Chinese",
            "text": "请在会议开始前完成设备检查，并把结果同步给我。",
            "instruct": "用严肃、坚定的声音说，重点词稍微加重。",
        },
        {
            "name": "en_bright",
            "language": "English",
            "text": "The test plan is ready, and we can start the next review tomorrow morning.",
            "instruct": "Speak in a bright energetic female voice, slightly excited and upbeat.",
        },
        {
            "name": "en_calm",
            "language": "English",
            "text": "There is no need to rush this; we can break the work into smaller steps.",
            "instruct": "Speak in a low calm male voice, slow pace, controlled emotion.",
        },
    ]

    if args.compare_base:
        print("Loading base model for comparison...")
        base_tts = Qwen3TTSModel.from_pretrained(
            args.base_model_path,
            device_map=args.device,
            dtype=dtype,
            attn_implementation="eager",
        )
        for sample in samples:
            generate_one(
                base_tts,
                sample["text"],
                sample["instruct"],
                sample["language"],
                os.path.join(args.output_dir, f"base_{sample['name']}.wav"),
                args.max_new_tokens,
            )
        del base_tts
        torch.cuda.empty_cache()

    print("Loading base model + LoRA adapter...")
    tuned_tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    tuned_tts = load_lora_checkpoint(
        tuned_tts,
        args.checkpoint_dir,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    for sample in samples:
        generate_one(
            tuned_tts,
            sample["text"],
            sample["instruct"],
            sample["language"],
            os.path.join(args.output_dir, f"lora_{sample['name']}.wav"),
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
