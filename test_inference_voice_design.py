import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    print("Loading model...")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")

    # -------- Single --------
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        language="Chinese",
        instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_voice_design_single.wav", wavs[0], sr)
    print("Saved to qwen3_tts_test_voice_design_single.wav")

    # -------- Batch --------
    texts = [
        "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
    ]
    languages = ["Chinese", "English"]
    instructs = [
        "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
    ]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=texts,
        language=languages,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"qwen3_tts_test_voice_design_batch_{i}.wav", w, sr)
    print("Batch generation complete!")


if __name__ == "__main__":
    main()
