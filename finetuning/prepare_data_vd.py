# coding=utf-8
"""Extract audio_codes from generated VoiceDesign dataset using the 12Hz tokenizer."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, default="train_raw.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="train_with_codes.jsonl")
    args = parser.parse_args()

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    all_lines = [json.loads(line.strip()) for line in open(args.input_jsonl, encoding="utf-8")]

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in all_lines:
        batch_lines.append(line)
        batch_audios.append(line["audio"])
        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer.encode(batch_audios)
            for code, ln in zip(enc_res.audio_codes, batch_lines):
                ln["audio_codes"] = code.cpu().tolist()
                final_lines.append(ln)
            batch_lines.clear()
            batch_audios.clear()

    if batch_audios:
        enc_res = tokenizer.encode(batch_audios)
        for code, ln in zip(enc_res.audio_codes, batch_lines):
            ln["audio_codes"] = code.cpu().tolist()
            final_lines.append(ln)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Prepared {len(final_lines)} samples -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
