# VoiceDesign Factorized Conditioning Path Audit

## Scope and checkpoint configuration

This audit covers the local Qwen3-TTS VoiceDesign inference path before training. The inspected local checkpoint is `Qwen3-TTS-12Hz-1.7B-VoiceDesign`: text embedding size 2048, Talker hidden size 2048, 28 Talker layers, codec vocabulary size 3072, and 16 codec groups. Shapes below use batch size `B`, instruction length `Ti`, target-text length `Tt`, Talker prefill length `Tp`, generated speech-code length `Ts`, and hidden size `D=2048`.

## Current path

1. **Instruction and target-text construction**
   - File: `qwen_tts/inference/qwen3_tts_model.py`
   - Functions: `Qwen3TTSModel._build_instruct_text`, `_build_assistant_text`, `generate_voice_design`
   - Legacy instruction becomes `<|im_start|>user\n{instruct}<|im_end|>\n`.
   - Target text independently becomes an assistant message followed by a second assistant prefix.
   - Before this change, `generate_voice_design` exposed only one `instruct` field for both voice and style.

2. **Tokenizer call**
   - File: `qwen_tts/inference/qwen3_tts_model.py`
   - Function: `Qwen3TTSModel._tokenize_texts`
   - `self.processor(text=..., return_tensors="pt", padding=True)` creates integer IDs `[1, Ti]` and `[1, Tt]` for each sample.
   - Factorized mode now uses `qwen_tts/inference/factorized_prompt.py::build_factorized_prompt` and piecewise `tokenizer.encode(..., add_special_tokens=False)` so the returned half-open spans align exactly with the returned token sequence.

3. **Text embeddings and Text Resize MLP**
   - File: `qwen_tts/core/models/modeling_qwen3_tts.py`
   - Classes/functions: `Qwen3TTSTalkerModel.get_text_embeddings`, `Qwen3TTSTalkerForConditionalGeneration.__init__`, `Qwen3TTSForConditionalGeneration.generate`
   - `text_embedding`: token IDs `[1, T]` -> text states `[1, T, 2048]`.
   - `text_projection` (`Qwen3TTSTalkerResizeMLP`) is the Text Resize MLP: `[1, T, text_hidden_size=2048]` -> `[1, T, talker_hidden_size=2048]` for the inspected 1.7B VoiceDesign checkpoint.
   - Instruction states were projected and appended directly to the per-sample Talker prefill list.

4. **Instruction/target merge and Talker prefill**
   - File: `qwen_tts/core/models/modeling_qwen3_tts.py`
   - Function: `Qwen3TTSForConditionalGeneration.generate`
   - Projected instruction `[1, Ti, D]` is the first optional prefill block.
   - Target-text role, TTS special tokens, codec prefill, and projected target text are then assembled as `talker_input_embed` and concatenated with instruction states.
   - Variable-length sequences are left padded to `[B, Tp, D]`, with attention mask `[B, Tp]`.
   - `self.talker.generate(inputs_embeds=..., attention_mask=..., trailing_text_hidden=...)` sends the final conditioning sequence into the main Talker.

5. **Talker input/output**
   - File: `qwen_tts/core/models/modeling_qwen3_tts.py`
   - Classes: `Qwen3TTSTalkerModel`, `Qwen3TTSTalkerForConditionalGeneration`
   - Main Talker consumes `inputs_embeds [B, Tp, 2048]` and an attention mask `[B, Tp]` during prefill, then autoregressively produces hidden states with last dimension 2048.
   - `codec_head` maps Talker states to first-codebook logits with vocabulary size 3072.
   - Generated first-codebook IDs and Code Predictor results form speech codes `[B, Ts, 16]` for the inspected 12 Hz checkpoint. Returned per-sample Talker hidden states have shape `[Ts, 2048]` after trimming.

6. **Code Predictor and waveform path**
   - `Qwen3TTSTalkerForConditionalGeneration.code_predictor` predicts the remaining codec groups from Talker states.
   - `Qwen3TTSModel.generate_voice_design` calls `self.model.speech_tokenizer.decode(...)` only after speech codes are complete.
   - The Code Predictor, codec head, speech tokenizer, speech-tokenizer decoder, and waveform decoder are downstream of the proposed conditioning interface and do not need structural changes in this stage.

## Adapter insertion

The insertion point is the projected instruction block in `Qwen3TTSForConditionalGeneration.generate`, immediately after:

```text
text token IDs -> text_embedding -> text_projection (Text Resize MLP)
```

and before that block is concatenated into Talker prefill. `instruct_segment_ids [1, Ti]` selects voice/style/text positions. `PromptSegmentEmbedding` and `VoiceStyleConditioningAdapter` preserve `[1, Ti, 2048]`. When the top-level configuration has `factorized_conditioning.enabled=false`, or when segment IDs are absent, `apply_factorized_conditioning` returns the original tensor object unchanged.

## Structured prompt span limitation

`build_factorized_prompt` tokenizes structural and content pieces independently and concatenates their token IDs. This provides exact spans for that returned piecewise token sequence and avoids silent alignment failure. It may not equal a separate, monolithic tokenizer call if a tokenizer merges across piece boundaries; inference therefore consumes the returned concatenated IDs directly.

Without a tokenizer, the utility explicitly reports `span_is_exact=false`, returns character-level approximate spans, and provides a warning. Such spans must not be passed to the model as token spans.

The current Talker architecture still requires the original assistant target-text continuation. Factorized mode also includes `<text>...</text>` inside structured instruction conditioning to establish a text segment, while retaining the original target-text path. This duplication is an explicit skeleton-stage limitation requiring ablation; it is not hidden behavior.

## Modules intentionally unchanged

- Main Talker decoder layers and pretrained weights
- Code Predictor and all codebook prediction logic
- Codec head
- Speech tokenizer and Speech Tokenizer Decoder
- Waveform decoder
- Speaker encoder / voice-clone path
- Checkpoint files

## Risks

- New tag tokens are ordinary tokenizer tokens unless a future tokenizer vocabulary explicitly reserves them.
- The adapter is untrained; zero initialization preserves its initial residual numerically, but the structured tagged prompt itself differs from the legacy prompt.
- Enabling the new modules adds missing parameters relative to an old checkpoint; those parameters require training and explicit checkpoint handling.
- Factorized and legacy samples must not be mixed ambiguously in one API call.
- Longer structured conditioning increases prefill length and memory.
- Text duplication and segment-boundary tokenization need ablation.
- Default-disabled behavior must be regression-tested against the legacy `instruct` path before any training work.

This stage does not modify Code Predictor, Speech Tokenizer Decoder, or waveform decoding.
