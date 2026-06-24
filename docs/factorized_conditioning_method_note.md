# Factorized Voice-Style Conditioning: Method Note

## Motivation

The original VoiceDesign interface represents speaker-related descriptions, timbral attributes, expressive style, and prosodic controls through a single natural-language instruction channel. This design is flexible, but the model receives no explicit variable indicating whether a token refers to a relatively stable voice property or a sample-specific speaking style. Consequently, supervision may permit voice and style cues to become entangled in the shared conditioning representation.

Prompt engineering can improve lexical regularity, for example by adding headings for voice and style. It does not, by itself, create a model-internal distinction: all prompt tokens still pass through the same embedding and projection path without typed segment information. Any apparent separation remains dependent on the pretrained model inferring the headings consistently.

## Structural interface

We introduce a factorized input interface with three typed segments:

```text
voice condition -> segment 1
style condition -> segment 2
target text     -> segment 3
special/tags    -> segment 0
```

The tagged representation is tokenized with explicit span accounting. After the existing Text Resize MLP, a learned segment embedding can be added, followed by independent lightweight residual bottleneck adapters for voice and style states. A text adapter is optional and disabled in the initial configuration. The residual up-projections and segment embeddings are zero-initialized, making the new modules an identity mapping at initialization.

The initial adapter is deliberately simple:

```text
A_s(h) = W_up,s SiLU(W_down,s h)
h'_s   = h_s + A_s(h_s)
```

where `s` is the voice, style, or optional text segment. Segment masks prevent a segment-specific adapter from modifying other positions.

## Internal and external factorization

The present work addresses **internal** factorization through typed prompt segments and segment-specific adapters. A future extension may add an **external** anchor speaker embedding derived from reference audio or a speaker encoder. That extension is not implemented here because it introduces an additional supervision source and would confound evaluation of the text-conditioning interface.

## Preserved components

The intervention occurs before the main Talker prefill. The Talker decoder, Code Predictor, codec head, Speech Tokenizer Decoder, and waveform decoder remain unchanged. These components model speech-code generation and waveform reconstruction downstream of the conditioning representation; modifying them is unnecessary for testing whether factorized textual conditioning improves controllability and would substantially enlarge the experimental search space.

## Required evaluation

The skeleton requires training before it can test the hypothesis. Subsequent experiments should include at least:

1. legacy mixed prompt baseline;
2. structured headings without segment embeddings;
3. segment embeddings only;
4. segment embeddings plus voice/style adapters;
5. optional text adapter;
6. voice/style swap and cross-combination generalization;
7. legacy-path regression and objective speaker/style/content metrics.

Evaluation must distinguish speaker consistency, style adherence, content correctness, and acoustic quality. An auxiliary multimodal judge cannot replace ECAPA/CAM++, ASR CER, or objective prosodic measurements.

This stage only implements the structural interface and adapter skeleton. It does not yet demonstrate improved controllability without training and evaluation.
