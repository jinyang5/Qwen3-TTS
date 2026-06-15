# VoiceDesign Finetune Notes

This note records the current VoiceDesign LoRA fine-tuning setup, known numerical issues, and the stable commands.

## Overview

VoiceDesign finetuning uses LoRA rank 8 on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of the main talker transformer. The talker has 28 layers and hidden size 2048. The `code_predictor` stays frozen. `text_projection`, a 2-layer MLP with about 8.4M parameters, can optionally be trained.

## Known Issues

### 1. Ill-Conditioned Pretrained Attention Projections

The released VoiceDesign checkpoint has extremely ill-conditioned attention projection matrices in problematic layers:

- `q_proj` condition numbers: over 100,000 in the worst sampled layers, with `sv_min` as low as about `2e-5`.
- `o_proj` condition numbers: over 100,000 in the worst sampled layers.
- `k_norm` weights: max `68.0` in layer 0, std `6.6`, much larger than the expected scale around `1.0`.

This causes Inf gradients during backward through frozen pretrained layers. These Inf values do not always break LoRA-only training because the trainable LoRA parameters can still have a usable gradient path. However, Inf gradients can become NaN through `Inf * 0 = NaN` when interacting with dropout-zeroed values or trainable operations.

### 2. Full-Layer LoRA Produces NaN

Training LoRA on all 28 layers, or on layers 0-17, produces NaN gradients. The NaN is generated inside each layer's attention/LoRA computation and cannot be intercepted by layer-boundary gradient clipping alone.

Current workaround:

```text
--lora_layer_start 18
```

This trains only layers 18-27 and is stable in both fp32 and bf16.

### 3. text_projection Training Is Now Supported

Previously, `--train_text_projection` caused NaN in all trainable parameters. The root cause was that Inf gradients from frozen layers propagated backward to the transformer input, then flowed into `text_projection`'s backward graph.

The current fix is a backward hook on `input_embeddings`:

```python
input_embeddings.register_hook(
    lambda g: torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
)
```

In `sft_voice_design.py`, this hook is installed automatically when `input_embeddings.requires_grad` is true. That covers `--train_text_projection`. It has been verified stable in fp32 and bf16 with `--lora_layer_start 18`.

## Stable Configurations

| Config | Status |
|--------|--------|
| `--lora_layer_start 18` default | Stable |
| `--lora_layer_start 18 --train_text_projection` | Stable with input hook |
| `--lora_layer_start 18 --model_dtype bfloat16 --mixed_precision bf16` | Stable |
| Full-layer LoRA, `--lora_layer_start 0` | NaN, unsolved |
| `--lora_layer_start 0 --train_text_projection` | NaN, unsolved |

## Recommended Training Commands

Run from:

```powershell
cd E:\project\Qwen3-TTS-main\finetuning
```

LoRA only:

```powershell
python sft_voice_design.py `
  --train_jsonl train_with_codes.jsonl `
  --init_model_path e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --output_model_path output_vd `
  --batch_size 1 `
  --num_epochs 5 `
  --lr 1e-5 `
  --model_dtype bfloat16 `
  --mixed_precision bf16
```

LoRA plus `text_projection`:

```powershell
python sft_voice_design.py `
  --train_jsonl train_with_codes.jsonl `
  --init_model_path e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --output_model_path output_vd `
  --batch_size 1 `
  --num_epochs 5 `
  --lr 1e-5 `
  --model_dtype bfloat16 `
  --mixed_precision bf16 `
  --train_text_projection
```

## Debug Scripts

Debug scripts are archived in `finetuning/debug/`:

- `debug_tp_compare.py` - Compare gradient norms with and without `text_projection` trainable.
- `debug_tp_anomaly.py` - Use `torch.autograd.detect_anomaly` to find NaN-producing operations.
- `debug_tp_backward.py` - Trace the first NaN occurrence in backward.
- `diagnose_weights.py` - Analyze RMSNorm weights, attention projection SVD, and condition numbers.

Because these files are now one level deeper under `finetuning/debug/`, their repo-root path setup must use `Path(__file__).resolve().parents[2]`.

## Tips

- `--skip_aux_codec` is for debugging only. It changes the input distribution and degrades loss, so do not use it for normal training.
- Gradient checkpointing is disabled because it can cause NaN in backward with bf16 + LoRA. The model fits 16GB without it at `batch_size=1`, bf16.
- `AdamW` uses `eps=1e-6` instead of the default `1e-8` for bf16 stability.
- Avoid `--lora_layer_start 0` for now. Full-layer LoRA still needs a deeper checkpoint-level fix, such as projection weight regularization or reinitialization.
