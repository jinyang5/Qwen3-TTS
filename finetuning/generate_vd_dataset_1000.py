# coding=utf-8
"""Generate a resumable 1000-sample VoiceDesign dataset.

Outputs:
  - data_1000/sample_0000.wav ...
  - train_raw_1000.jsonl with audio/text/instruct/language fields

Run from Qwen3-TTS-main/finetuning.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel


ZH_TEXT_TEMPLATES = [
    "{person}今天需要确认{task}，请在{time}之前给我一个明确答复。",
    "如果{event}进展顺利，我们就可以开始准备{next_step}。",
    "我刚刚看完{topic}的资料，里面有几个细节值得再讨论。",
    "{place}的天气变化很快，出门前最好再检查一下行程。",
    "这件事情不用着急，我们先把{task}拆成几个简单步骤。",
    "请把{object}放到桌子左边，我稍后会统一整理。",
    "听到这个消息之后，我心里终于踏实了一些。",
    "虽然结果还没有出来，但我们已经比昨天更接近目标了。",
    "麻烦你重复一遍刚才的问题，我想确认自己没有听错。",
    "晚上的会议改到线上进行，链接会提前十分钟发出。",
]

EN_TEXT_TEMPLATES = [
    "{person} needs a clear update about {task} before {time}.",
    "If {event} goes well, we can start preparing {next_step}.",
    "I just reviewed the notes about {topic}, and a few details need attention.",
    "The weather near {place} changes quickly, so check the route before leaving.",
    "There is no need to rush this; we can break {task} into smaller steps.",
    "Please place the {object} on the left side of the desk for now.",
    "After hearing the news, I finally felt a little more at ease.",
    "The result is not final yet, but we are closer than we were yesterday.",
    "Could you repeat the question once more so I can make sure I understood it?",
    "Tonight's meeting will move online, and the link will be sent ten minutes early.",
]

ZH_INSTRUCTS = [
    "用温柔、自然的年轻女声说，语速中等，语气亲切。",
    "用沉稳、清晰的中年男声说，语气可靠，节奏平稳。",
    "用活泼、明亮的女声说，语气轻快，带一点期待感。",
    "用低沉、冷静的男声说，语速稍慢，情绪克制。",
    "用正式、专业的播报声说，咬字清楚，停顿自然。",
    "用疲惫但温和的声音说，语速偏慢，情绪真实。",
    "用开心、兴奋的年轻声音说，语调上扬，节奏稍快。",
    "用安静、柔和的旁白声说，像在讲一个温暖的故事。",
    "用严肃、坚定的声音说，重点词稍微加重。",
    "用犹豫、困惑的声音说，语调有轻微起伏。",
]

EN_INSTRUCTS = [
    "Speak in a warm young female voice, natural pace, friendly tone.",
    "Speak in a calm middle-aged male voice, steady and trustworthy.",
    "Speak in a bright energetic female voice, slightly excited and upbeat.",
    "Speak in a low calm male voice, slow pace, controlled emotion.",
    "Speak in a professional announcer voice, clear articulation and natural pauses.",
    "Speak in a tired but gentle voice, slower pace, realistic emotion.",
    "Speak in a happy young voice, rising intonation, slightly fast pace.",
    "Speak in a quiet soft narrator voice, like telling a warm story.",
    "Speak in a serious firm voice, emphasizing important words.",
    "Speak in a hesitant confused voice, with subtle pitch variation.",
]

ZH_SLOTS = {
    "person": ["小林", "项目经理", "客服同事", "设计师", "测试同事", "值班主管"],
    "task": ["预算调整", "测试计划", "交付时间", "样本整理", "会议纪要", "设备检查"],
    "time": ["下午三点", "明天上午", "周五之前", "今晚八点", "下班之前", "下周一"],
    "event": ["系统升级", "客户确认", "样机测试", "数据同步", "场地布置", "版本发布"],
    "next_step": ["正式发布", "数据复核", "下一轮评审", "资料归档", "排期确认", "现场测试"],
    "topic": ["用户反馈", "市场报告", "训练日志", "质量检查", "产品方案", "语音样本"],
    "place": ["机场附近", "会议中心", "老城区", "办公楼南门", "展厅入口", "录音室"],
    "object": ["蓝色文件夹", "录音设备", "备用钥匙", "白色纸箱", "会议平板", "样品标签"],
}

EN_SLOTS = {
    "person": ["The team lead", "The designer", "My colleague", "The support agent", "The project manager"],
    "task": ["budget update", "test plan", "delivery schedule", "sample review", "meeting notes", "device check"],
    "time": ["3 p.m.", "tomorrow morning", "Friday", "8 tonight", "the end of the day", "next Monday"],
    "event": ["the system update", "client approval", "prototype test", "data sync", "venue setup", "version release"],
    "next_step": ["the public release", "data review", "the next review", "document archive", "schedule confirmation"],
    "topic": ["user feedback", "market report", "training logs", "quality check", "product proposal", "voice samples"],
    "place": ["the airport", "the conference center", "the old district", "the south gate", "the showroom entrance"],
    "object": ["blue folder", "recording device", "spare key", "white box", "meeting tablet", "sample label"],
}


def fill_template(template: str, rng: random.Random, slots: dict[str, list[str]]) -> str:
    values = {key: rng.choice(value) for key, value in slots.items()}
    return template.format(**values)


def build_samples(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        use_zh = i % 2 == 0
        if use_zh:
            text = fill_template(rng.choice(ZH_TEXT_TEMPLATES), rng, ZH_SLOTS)
            instruct = rng.choice(ZH_INSTRUCTS)
            language = "Chinese"
        else:
            text = fill_template(rng.choice(EN_TEXT_TEMPLATES), rng, EN_SLOTS)
            instruct = rng.choice(EN_INSTRUCTS)
            language = "English"
        rows.append({"text": text, "instruct": instruct, "language": language})
    return rows


def load_done(output_jsonl: str) -> set[int]:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stem = Path(row["audio"]).stem
            if stem.startswith("sample_"):
                done.add(int(stem.split("_", 1)[1]))
    return done


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output_dir", type=str, default="data_1000")
    parser.add_argument("--output_jsonl", type=str, default="train_raw_1000.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples = build_samples(args.count, args.seed)
    done = load_done(args.output_jsonl)

    print(f"Loading VoiceDesign model from {args.model_path} ...", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"Model loaded. Existing completed samples: {len(done)}", flush=True)

    for i, sample in enumerate(samples):
        audio_path = os.path.join(args.output_dir, f"sample_{i:04d}.wav")
        if i in done and os.path.exists(audio_path):
            continue

        print(f"[{i + 1}/{args.count}] {sample['language']} | {sample['text']}", flush=True)
        start = time.time()
        last_error = None
        for attempt in range(args.retries + 1):
            try:
                wavs, sr = tts.generate_voice_design(
                    text=sample["text"],
                    language=sample["language"],
                    instruct=sample["instruct"],
                    max_new_tokens=args.max_new_tokens,
                )
                sf.write(audio_path, wavs[0], sr)
                row = {
                    "audio": audio_path.replace("\\", "/"),
                    "text": sample["text"],
                    "instruct": sample["instruct"],
                    "language": sample["language"],
                }
                append_jsonl(args.output_jsonl, row)
                print(f"  saved {audio_path} in {time.time() - start:.1f}s", flush=True)
                break
            except Exception as exc:
                last_error = exc
                print(f"  attempt {attempt + 1} failed: {exc}", flush=True)
                torch.cuda.empty_cache()
                time.sleep(2)
        else:
            print(f"  failed permanently: {last_error}", flush=True)

        torch.cuda.empty_cache()

    print(f"Done. JSONL: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
