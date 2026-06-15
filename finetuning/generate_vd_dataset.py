# coding=utf-8
"""Generate a small test dataset for VoiceDesign fine-tuning using the model itself."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel

SAMPLES = [
    # (text, instruct, language)
    ("今天天气真好，我们出去散步吧。", "用温柔、愉快的年轻女声说，语气轻快，带一点俏皮的感觉。", "Chinese"),
    ("我对这件事感到非常愤怒，请立刻解决这个问题。", "用低沉、严厉的中年男声说，语气坚定而愤怒，语速偏快。", "Chinese"),
    ("晚安，做个好梦，明天又是充满希望的一天。", "用温暖、柔和的女声说，语速缓慢，像是在哄小孩入睡。", "Chinese"),
    ("I'm absolutely thrilled to announce our new product launch!", "Speak in an energetic, bright young female voice, high pitch, excited and enthusiastic tone.", "English"),
    ("I'm deeply disappointed with the outcome of this project.", "Speak in a calm but sad male voice, low pitch, slow pace, with a sense of resignation.", "English"),
    ("The ancient castle stood silently on the hill, watching centuries pass.", "Speak in a deep, mysterious male narrator voice, slow and dramatic, like a documentary.", "English"),
    ("快点快点，我们要迟到了！", "用急促、焦虑的年轻男声说，语速快，声音偏高，带着紧张感。", "Chinese"),
    ("让我们一起为这个伟大的成就鼓掌吧！", "用洪亮、庄重的男声说，语速中等，像是在做正式演讲。", "Chinese"),
    ("Darling, I've been waiting for you all day long.", "Speak in a soft, romantic female voice, gentle and intimate, slow pace with a slight whisper quality.", "English"),
    ("说实话，我真的不知道该怎么办才好。", "用犹豫、困惑的年轻女声说，语速较慢，音调有起伏，带着不确定感。", "Chinese"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_jsonl", type=str, default="train_raw.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading VoiceDesign model from {args.model_path} ...")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
    )
    print("Model loaded.")

    rows = []
    for i, (text, instruct, lang) in enumerate(SAMPLES):
        print(f"\n[{i+1}/{len(SAMPLES)}] Generating...")
        print(f"  Text: {text[:50]}...")
        print(f"  Instruct: {instruct[:50]}...")

        t0 = time.time()
        wavs, sr = tts.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct,
            max_new_tokens=2048,
        )
        elapsed = time.time() - t0

        audio_path = os.path.join(args.output_dir, f"sample_{i:03d}.wav")
        sf.write(audio_path, wavs[0], sr)
        print(f"  Saved: {audio_path}  ({elapsed:.1f}s)")

        rows.append({
            "audio": audio_path,
            "text": text,
            "instruct": instruct,
            "language": lang,
        })

        # Free GPU cache
        torch.cuda.empty_cache()

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to {args.output_jsonl} ({len(rows)} samples)")
    print("Next: run prepare_data_vd.py to extract audio codes.")


if __name__ == "__main__":
    main()
