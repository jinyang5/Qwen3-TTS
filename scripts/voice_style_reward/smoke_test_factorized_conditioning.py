#!/usr/bin/env python3
"""CPU-only smoke test for the factorized conditioning interface and modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_tts.core.models.adapters.factorized_conditioning import (  # noqa: E402
    PromptSegmentEmbedding,
    VoiceStyleConditioningAdapter,
)
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration  # noqa: E402
from qwen_tts.inference.factorized_prompt import build_factorized_prompt  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _CharacterTokenizer()

    def __call__(self, text: str, return_tensors: str, padding: bool) -> dict[str, torch.Tensor]:
        del return_tensors, padding
        return {"input_ids": torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)}


class _FakeSpeechTokenizer:
    def decode(self, codes):
        return [np.zeros(1, dtype=np.float32) for _ in codes], 24000


class _FakeModel:
    tts_model_type = "voice_design"
    tokenizer_type = "fake"
    tts_model_size = "smoke"
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.speech_tokenizer = _FakeSpeechTokenizer()
        self.last_generate_kwargs = None

    def get_supported_languages(self):
        return ["auto", "chinese", "english"]

    def get_supported_speakers(self):
        return []

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        batch_size = len(kwargs["input_ids"])
        return [torch.zeros((1, 16), dtype=torch.long) for _ in range(batch_size)], None


def check(name: str, condition: bool, details: dict | None = None) -> dict:
    if not condition:
        raise AssertionError(name)
    return {"name": name, "passed": True, "details": details or {}}


def main() -> None:
    checks = []
    tokenizer = _CharacterTokenizer()
    structured = build_factorized_prompt(
        voice_prompt="一位年轻女性说话人，音色明亮清晰。",
        style_prompt="开心明亮地说，语速自然略快。",
        text="今天我们介绍一项新的语音实验。",
        tokenizer=tokenizer,
    )
    checks.append(check("factorized_full_prompt", "<voice>" in structured["full_prompt"] and "<style>" in structured["full_prompt"] and "<text>" in structured["full_prompt"]))
    checks.append(check("factorized_token_spans_exact", structured["span_is_exact"]))
    checks.append(check("segment_ids_include_voice_style_text", {1, 2, 3}.issubset(set(structured["segment_ids"]))))
    checks.append(check("segment_ids_align_with_tokens", len(structured["segment_ids"]) == len(structured["input_ids"])))

    hidden_states = torch.randn(2, 13, 2048)
    segment_ids = torch.tensor(
        [[0, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0]] * 2,
        dtype=torch.long,
    )
    segment_embedding = PromptSegmentEmbedding(hidden_size=2048, num_segments=4)
    adapter = VoiceStyleConditioningAdapter(hidden_size=2048, bottleneck_size=256)
    after_segments = segment_embedding(hidden_states, segment_ids)
    adapted = adapter(after_segments, segment_ids)
    max_initial_difference = float((adapted - hidden_states).abs().max())
    checks.append(check("segment_embedding_shape", after_segments.shape == hidden_states.shape))
    checks.append(check("adapter_shape", adapted.shape == hidden_states.shape))
    checks.append(check("adapter_near_identity_initialization", max_initial_difference <= 1e-7, {"max_abs_difference": max_initial_difference}))
    checks.append(check("adapter_bypass_exact", torch.equal(adapter(hidden_states, segment_ids, enabled=False), hidden_states)))
    integration_stub = SimpleNamespace(
        factorized_conditioning_enabled=True,
        factorized_segment_embedding=segment_embedding,
        voice_style_conditioning_adapter=adapter,
    )
    integrated = Qwen3TTSForConditionalGeneration.apply_factorized_conditioning(
        integration_stub, hidden_states, segment_ids
    )
    checks.append(check("model_insertion_helper_shape", integrated.shape == hidden_states.shape))
    integration_stub.factorized_conditioning_enabled = False
    bypassed = Qwen3TTSForConditionalGeneration.apply_factorized_conditioning(
        integration_stub, hidden_states, segment_ids
    )
    checks.append(check("model_insertion_disabled_exact", bypassed is hidden_states))
    init_stub = SimpleNamespace(config=SimpleNamespace(initializer_range=0.02))
    segment_embedding.apply(
        lambda module: Qwen3TTSForConditionalGeneration._init_weights(init_stub, module)
    )
    adapter.apply(
        lambda module: Qwen3TTSForConditionalGeneration._init_weights(init_stub, module)
    )
    reinitialized = adapter(segment_embedding(hidden_states, segment_ids), segment_ids)
    checks.append(
        check(
            "identity_survives_model_missing_key_initialization",
            float((reinitialized - hidden_states).abs().max()) <= 1e-7,
        )
    )

    fake_model = _FakeModel()
    wrapper = Qwen3TTSModel(fake_model, _FakeProcessor())
    legacy_instruction = "一位年轻女性，声音明亮活泼，用开心的语气说。"
    wrapper.generate_voice_design(
        text="今天我们介绍一项新的语音实验。",
        instruct=legacy_instruction,
        language="zh",
    )
    legacy_kwargs = fake_model.last_generate_kwargs
    expected_legacy_ids = tokenizer.encode(wrapper._build_instruct_text(legacy_instruction))
    checks.append(check("legacy_mixed_has_no_segment_ids", legacy_kwargs["instruct_segment_ids"] is None))
    checks.append(check("legacy_mixed_tokenization_unchanged", legacy_kwargs["instruct_ids"][0].squeeze(0).tolist() == expected_legacy_ids))
    codec_return = wrapper.generate_voice_design(
        text="今天我们介绍一项新的语音实验。",
        instruct=legacy_instruction,
        language="zh",
        return_codec_codes=True,
    )
    checks.append(
        check(
            "optional_codec_return_preserves_default_api",
            len(codec_return) == 3 and codec_return[2][0].shape == (1, 16),
        )
    )

    wrapper.generate_voice_design(
        text="今天我们介绍一项新的语音实验。",
        voice_prompt="一位年轻女性说话人，音色明亮清晰。",
        style_prompt="开心明亮地说，语速自然略快。",
        language="zh",
        prompt_format="factorized",
    )
    factorized_kwargs = fake_model.last_generate_kwargs
    model_segment_ids = factorized_kwargs["instruct_segment_ids"][0]
    checks.append(check("factorized_wrapper_returns_segment_ids", model_segment_ids is not None))
    checks.append(check("factorized_wrapper_segments_include_all_types", {1, 2, 3}.issubset(set(model_segment_ids.flatten().tolist()))))
    checks.append(check("factorized_wrapper_ids_align", model_segment_ids.shape == factorized_kwargs["instruct_ids"][0].shape))

    default_config = Qwen3TTSConfig()
    checks.append(check("factorized_config_disabled_by_default", default_config.factorized_conditioning["enabled"] is False))
    report = {
        "passed": all(item["passed"] for item in checks),
        "num_checks": len(checks),
        "num_passed": sum(item["passed"] for item in checks),
        "loaded_tts_checkpoint": False,
        "generated_audio": False,
        "checks": checks,
    }
    output = ROOT / "outputs/factorized_conditioning/smoke_test_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
