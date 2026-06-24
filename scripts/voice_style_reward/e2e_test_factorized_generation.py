#!/usr/bin/env python3
"""Real-checkpoint E2E regression and codec-target feasibility audit."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_tts import Qwen3TTSModel  # noqa: E402
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig  # noqa: E402
from qwen_tts.inference.factorized_prompt import build_factorized_prompt  # noqa: E402


TEXT = "今天我们介绍一项新的语音实验。"
MIXED_INSTRUCT = "一位年轻女性说话人，普通话标准，音色明亮清晰，声线偏轻，用开心明亮的语气说。"
VOICE_PROMPT = "一位年轻女性说话人，普通话标准，音色明亮清晰，声线偏轻，音高略高。"
HAPPY_STYLE = "开心明亮地说，语气积极，能量中等偏高，语速自然略快。"
SLOW_STYLE = "清晰自然地慢速说出，语速明显放慢，停顿更充分，但不要拖音或长时间静音。"
SEED = 20260625

MANIFEST_FIELDS = [
    "sample_id",
    "mode",
    "factorized_conditioning_enabled",
    "instruct",
    "voice_prompt",
    "style_prompt",
    "text",
    "language",
    "full_prompt",
    "voice_span",
    "style_span",
    "text_span",
    "segment_ids_present",
    "segment_types",
    "segment_embedding_invoked",
    "adapter_invoked",
    "wav_path",
    "duration_sec",
    "rms",
    "peak",
    "near_silent",
    "codec_codes_available",
    "codec_shape",
    "codec_path",
    "generation_seconds",
    "generation_error",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audio_metrics(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    samples = np.asarray(wav, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return {
        "duration_sec": float(samples.size / sample_rate),
        "rms": rms,
        "peak": peak,
        "near_silent": rms < 1e-4,
    }


def load_model(model_path: Path, factorized_enabled: bool) -> tuple[Qwen3TTSModel, float]:
    config = Qwen3TTSConfig.from_pretrained(model_path)
    config.factorized_conditioning = {
        **config.factorized_conditioning,
        "enabled": factorized_enabled,
    }
    started = time.time()
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        config=config,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    return model, time.time() - started


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def structured_metadata(model: Qwen3TTSModel, style_prompt: str) -> dict[str, Any]:
    tokenizer = getattr(model.processor, "tokenizer", model.processor)
    structured = build_factorized_prompt(
        voice_prompt=VOICE_PROMPT,
        style_prompt=style_prompt,
        text=TEXT,
        tokenizer=tokenizer,
    )
    return {
        "full_prompt": structured["full_prompt"],
        "voice_span": structured["voice_span"],
        "style_span": structured["style_span"],
        "text_span": structured["text_span"],
        "segment_ids_present": True,
        "segment_types": sorted(set(structured["segment_ids"])),
        "num_prompt_tokens": len(structured["input_ids"]),
    }


def save_generation(
    model: Qwen3TTSModel,
    output_dir: Path,
    sample_id: str,
    mode: str,
    factorized_enabled: bool,
    style_prompt: str = "",
    invocation_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], torch.Tensor | None, np.ndarray | None, int | None]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "mode": mode,
        "factorized_conditioning_enabled": factorized_enabled,
        "instruct": MIXED_INSTRUCT if mode == "mixed" else "",
        "voice_prompt": VOICE_PROMPT if mode == "factorized" else "",
        "style_prompt": style_prompt if mode == "factorized" else "",
        "text": TEXT,
        "language": "zh",
        "full_prompt": "",
        "voice_span": "",
        "style_span": "",
        "text_span": "",
        "segment_ids_present": False,
        "segment_types": "",
        "segment_embedding_invoked": False,
        "adapter_invoked": False,
        "wav_path": "",
        "duration_sec": "",
        "rms": "",
        "peak": "",
        "near_silent": "",
        "codec_codes_available": False,
        "codec_shape": "",
        "codec_path": "",
        "generation_seconds": "",
        "generation_error": "",
    }
    if mode == "factorized":
        metadata = structured_metadata(model, style_prompt)
        row.update(
            {
                "full_prompt": metadata["full_prompt"],
                "voice_span": json.dumps(metadata["voice_span"]),
                "style_span": json.dumps(metadata["style_span"]),
                "text_span": json.dumps(metadata["text_span"]),
                "segment_ids_present": metadata["segment_ids_present"],
                "segment_types": json.dumps(metadata["segment_types"]),
            }
        )
    codes_cpu = None
    wav = None
    sample_rate = None
    try:
        set_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        common_kwargs = {
            "text": TEXT,
            "language": "zh",
            "do_sample": False,
            "subtalker_dosample": False,
            "max_new_tokens": 256,
            "return_codec_codes": True,
        }
        if mode == "mixed":
            wavs, sample_rate, codes = model.generate_voice_design(
                instruct=MIXED_INSTRUCT,
                **common_kwargs,
            )
        else:
            wavs, sample_rate, codes = model.generate_voice_design(
                voice_prompt=VOICE_PROMPT,
                style_prompt=style_prompt,
                prompt_format="factorized",
                **common_kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        row["generation_seconds"] = time.time() - started
        wav = np.asarray(wavs[0], dtype=np.float32)
        codes_cpu = codes[0].detach().to(torch.long).cpu()
        wav_path = output_dir / "wavs" / f"{sample_id}.wav"
        code_path = output_dir / "codes" / f"{sample_id}.npy"
        sf.write(wav_path, wav, sample_rate)
        np.save(code_path, codes_cpu.numpy())
        row.update(audio_metrics(wav, sample_rate))
        row.update(
            {
                "wav_path": str(wav_path),
                "codec_codes_available": codes_cpu.ndim == 2,
                "codec_shape": json.dumps(list(codes_cpu.shape)),
                "codec_path": str(code_path),
            }
        )
    except Exception as exc:
        row["generation_error"] = f"{type(exc).__name__}: {exc}"
    counts = invocation_counts or {}
    row["segment_embedding_invoked"] = counts.get("segment_embedding", 0) > 0
    row["adapter_invoked"] = counts.get("adapter", 0) > 0
    return row, codes_cpu, wav, sample_rate


def adapter_identity_state(model: Qwen3TTSModel) -> dict[str, Any]:
    core = model.model
    segment_module = core.factorized_segment_embedding
    adapter = core.voice_style_conditioning_adapter
    values = {
        "segment_embedding_max_abs": (
            float(segment_module.embedding.weight.detach().abs().max().cpu())
            if segment_module is not None
            else None
        ),
        "voice_up_weight_max_abs": (
            float(adapter.voice_adapter.up.weight.detach().abs().max().cpu())
            if adapter is not None
            else None
        ),
        "style_up_weight_max_abs": (
            float(adapter.style_adapter.up.weight.detach().abs().max().cpu())
            if adapter is not None
            else None
        ),
    }
    values["all_identity_parameters_zero"] = all(
        value == 0.0 for value in values.values() if isinstance(value, float)
    )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT.parent / "models/qwen3tts/VoiceDesign",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/factorized_conditioning_e2e",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real 1.7B checkpoint E2E test")
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"checkpoint directory not found: {model_path}")
    core_outputs = [
        output_dir / "mixed_regression_manifest.csv",
        output_dir / "factorized_generation_manifest.csv",
        output_dir / "zero_init_equivalence_report.json",
        output_dir / "codec_target_availability_report.json",
        output_dir / "e2e_summary.json",
    ]
    if not args.overwrite and any(path.exists() for path in core_outputs):
        raise SystemExit("E2E outputs already exist; use a new --output-dir or explicit --overwrite")
    for directory in (output_dir, output_dir / "wavs", output_dir / "codes"):
        directory.mkdir(parents=True, exist_ok=True)

    mixed_rows: list[dict[str, Any]] = []
    factorized_rows: list[dict[str, Any]] = []
    direct_codes: dict[str, torch.Tensor] = {}
    generated_wavs: dict[str, tuple[np.ndarray, int]] = {}
    load_seconds: dict[str, float] = {}

    print("Loading checkpoint with factorized_conditioning.enabled=false", flush=True)
    disabled_model, load_seconds["disabled"] = load_model(model_path, False)
    disabled_config = disabled_model.model.config.factorized_conditioning.copy()
    mixed_row, codes, wav, sr = save_generation(
        disabled_model, output_dir, "mixed_legacy", "mixed", False
    )
    mixed_rows.append(mixed_row)
    if codes is not None:
        direct_codes["mixed_legacy"] = codes
    if wav is not None and sr is not None:
        generated_wavs["mixed_legacy"] = (wav, sr)
    print(f"mixed: error={mixed_row['generation_error']!r} shape={mixed_row['codec_shape']}", flush=True)

    for sample_id, style_prompt in (
        ("factorized_happy_disabled", HAPPY_STYLE),
        ("factorized_slow_disabled", SLOW_STYLE),
    ):
        row, codes, wav, sr = save_generation(
            disabled_model, output_dir, sample_id, "factorized", False, style_prompt
        )
        factorized_rows.append(row)
        if codes is not None:
            direct_codes[sample_id] = codes
        if wav is not None and sr is not None:
            generated_wavs[sample_id] = (wav, sr)
        print(f"{sample_id}: error={row['generation_error']!r} shape={row['codec_shape']}", flush=True)

    encoder_audit: dict[str, Any] = {
        "attempted": False,
        "available": False,
        "shape": None,
        "path": None,
        "error": None,
    }
    encoder_source = generated_wavs.get("factorized_happy_disabled")
    if encoder_source is not None:
        encoder_audit["attempted"] = True
        try:
            encoded = disabled_model.model.speech_tokenizer.encode(
                encoder_source[0], sr=encoder_source[1]
            )
            encoded_codes = encoded.audio_codes[0].detach().to(torch.long).cpu()
            encoded_path = output_dir / "codes/factorized_happy_reencoded.npy"
            np.save(encoded_path, encoded_codes.numpy())
            encoder_audit.update(
                {
                    "available": encoded_codes.ndim == 2,
                    "shape": list(encoded_codes.shape),
                    "path": str(encoded_path),
                }
            )
            original_codes = direct_codes.get("factorized_happy_disabled")
            if original_codes is not None and original_codes.shape == encoded_codes.shape:
                matches = original_codes == encoded_codes
                encoder_audit.update(
                    {
                        "exactly_matches_direct_codes": bool(torch.equal(original_codes, encoded_codes)),
                        "token_agreement_rate": float(matches.float().mean()),
                        "token_difference_count": int((~matches).sum()),
                        "num_compared_tokens": int(matches.numel()),
                    }
                )
        except Exception as exc:
            encoder_audit["error"] = f"{type(exc).__name__}: {exc}"

    disabled_happy_codes = direct_codes.get("factorized_happy_disabled")
    disabled_happy_wav = generated_wavs.get("factorized_happy_disabled")
    del disabled_model
    release_cuda_memory()

    print("Loading checkpoint with factorized_conditioning.enabled=true", flush=True)
    enabled_model, load_seconds["enabled"] = load_model(model_path, True)
    identity_state = adapter_identity_state(enabled_model)
    invocation_counts = {"segment_embedding": 0, "adapter": 0}
    hooks = []
    if enabled_model.model.factorized_segment_embedding is not None:
        hooks.append(
            enabled_model.model.factorized_segment_embedding.register_forward_hook(
                lambda *_: invocation_counts.__setitem__(
                    "segment_embedding", invocation_counts["segment_embedding"] + 1
                )
            )
        )
    if enabled_model.model.voice_style_conditioning_adapter is not None:
        hooks.append(
            enabled_model.model.voice_style_conditioning_adapter.register_forward_hook(
                lambda *_: invocation_counts.__setitem__(
                    "adapter", invocation_counts["adapter"] + 1
                )
            )
        )
    enabled_row, enabled_codes, enabled_wav, enabled_sr = save_generation(
        enabled_model,
        output_dir,
        "factorized_happy_enabled",
        "factorized",
        True,
        HAPPY_STYLE,
        invocation_counts,
    )
    factorized_rows.append(enabled_row)
    for hook in hooks:
        hook.remove()
    print(f"factorized_happy_enabled: error={enabled_row['generation_error']!r} shape={enabled_row['codec_shape']}", flush=True)

    codes_equal = None
    code_difference_count = None
    if disabled_happy_codes is not None and enabled_codes is not None:
        codes_equal = bool(
            disabled_happy_codes.shape == enabled_codes.shape
            and torch.equal(disabled_happy_codes, enabled_codes)
        )
        if disabled_happy_codes.shape == enabled_codes.shape:
            code_difference_count = int((disabled_happy_codes != enabled_codes).sum())
    wav_max_abs_difference = None
    wav_rmse = None
    wav_relative_rmse = None
    wav_lengths_equal = None
    if disabled_happy_wav is not None and enabled_wav is not None:
        wav_lengths_equal = len(disabled_happy_wav[0]) == len(enabled_wav)
        if wav_lengths_equal:
            difference = disabled_happy_wav[0].astype(np.float64) - enabled_wav.astype(np.float64)
            wav_max_abs_difference = float(np.max(np.abs(difference)))
            wav_rmse = float(np.sqrt(np.mean(np.square(difference))))
            reference_rms = audio_metrics(disabled_happy_wav[0], disabled_happy_wav[1])["rms"]
            wav_relative_rmse = wav_rmse / max(reference_rms, 1e-12)
    equivalence_pass = bool(
        identity_state["all_identity_parameters_zero"]
        and invocation_counts["segment_embedding"] > 0
        and invocation_counts["adapter"] > 0
        and codes_equal is True
        and wav_lengths_equal is True
        and wav_relative_rmse is not None
        and wav_relative_rmse <= 0.01
    )
    zero_report = {
        "checkpoint": str(model_path),
        "seed": SEED,
        "generation_mode": "greedy",
        "same_full_prompt": enabled_row["full_prompt"] == factorized_rows[0]["full_prompt"],
        "full_prompt_sha256": hashlib.sha256(enabled_row["full_prompt"].encode()).hexdigest(),
        "disabled_config": disabled_config,
        "enabled_config": enabled_model.model.config.factorized_conditioning,
        "identity_parameter_state": identity_state,
        "segment_embedding_invocations": invocation_counts["segment_embedding"],
        "adapter_invocations": invocation_counts["adapter"],
        "disabled_codec_shape": list(disabled_happy_codes.shape) if disabled_happy_codes is not None else None,
        "enabled_codec_shape": list(enabled_codes.shape) if enabled_codes is not None else None,
        "codec_codes_exactly_equal": codes_equal,
        "codec_difference_count": code_difference_count,
        "wav_lengths_equal": wav_lengths_equal,
        "wav_max_abs_difference": wav_max_abs_difference,
        "wav_rmse": wav_rmse,
        "wav_relative_rmse": wav_relative_rmse,
        "equivalence_criterion": (
            "identity parameters are zero; adapters are invoked; generated codec matrices are exactly equal; "
            "waveform lengths match; waveform RMSE/reference RMS <= 1%"
        ),
        "equivalence_pass": equivalence_pass,
        "enabled_generation_error": enabled_row["generation_error"] or None,
    }
    del enabled_model
    release_cuda_memory()

    all_direct_shapes = {
        sample_id: list(codes.shape) for sample_id, codes in direct_codes.items()
    }
    direct_available = bool(all_direct_shapes) and all(
        len(shape) == 2 and shape[1] == 16 for shape in all_direct_shapes.values()
    )
    codec_report = {
        "direct_generation_codes": {
            "available": direct_available,
            "expected_shape": "[T, 16]",
            "sample_shapes": all_direct_shapes,
            "preferred_for_training_targets": direct_available,
            "note": "Direct Talker codes avoid waveform decode/re-encode information loss.",
        },
        "speech_tokenizer_encoder_fallback": encoder_audit,
        "training_data_resolution": (
            "Save direct generation codec matrices alongside each candidate WAV. "
            "Use speech_tokenizer.encode only for legacy WAVs without saved generation codes; "
            "re-encoded targets are not assumed token-identical to direct generation codes."
            if direct_available
            else "Use speech_tokenizer.encode(wav, sr) to recover codec targets and audit reconstruction quality."
        ),
    }
    write_csv(output_dir / "mixed_regression_manifest.csv", mixed_rows)
    write_csv(output_dir / "factorized_generation_manifest.csv", factorized_rows)
    write_json(output_dir / "zero_init_equivalence_report.json", zero_report)
    write_json(output_dir / "codec_target_availability_report.json", codec_report)

    mixed_ok = bool(mixed_rows and not mixed_rows[0]["generation_error"] and Path(mixed_rows[0]["wav_path"]).is_file())
    factorized_disabled_rows = [
        row for row in factorized_rows if not row["factorized_conditioning_enabled"]
    ]
    factorized_ok = bool(factorized_disabled_rows) and all(
        not row["generation_error"]
        and Path(row["wav_path"]).is_file()
        and row["segment_ids_present"]
        for row in factorized_disabled_rows
    )
    all_rows = mixed_rows + factorized_rows
    summary = {
        "checkpoint": str(model_path),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "load_seconds": load_seconds,
        "num_generations": len(all_rows),
        "num_success": sum(not row["generation_error"] for row in all_rows),
        "num_error": sum(bool(row["generation_error"]) for row in all_rows),
        "mixed_path_ok": mixed_ok,
        "factorized_path_ok": factorized_ok,
        "zero_init_equivalence_ok": equivalence_pass,
        "direct_codec_targets_available": direct_available,
        "speech_tokenizer_encoder_available": encoder_audit["available"],
        "near_silent_outputs": [row["sample_id"] for row in all_rows if row["near_silent"] is True],
        "training_performed": False,
        "omni_api_called": False,
        "checkpoint_modified": False,
        "ready_for_clean_pilot_generation": bool(
            mixed_ok and factorized_ok and equivalence_pass and direct_available
        ),
    }
    write_json(output_dir / "e2e_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["ready_for_clean_pilot_generation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
