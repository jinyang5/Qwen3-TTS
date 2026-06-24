"""Structured prompt construction for factorized voice/style conditioning."""

from __future__ import annotations

from typing import Any


VOICE_SEGMENT_ID = 1
STYLE_SEGMENT_ID = 2
TEXT_SEGMENT_ID = 3


def encode_text_piece(tokenizer: Any, text: str) -> list[int]:
    """Tokenize one prompt piece without tokenizer-added special tokens."""
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    elif callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    else:
        raise TypeError("tokenizer must expose encode() or be callable")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("expected a single tokenized sequence")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_factorized_prompt(
    voice_prompt: str,
    style_prompt: str,
    text: str,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build a tagged prompt and aligned half-open token spans.

    With a tokenizer, each structural/content piece is tokenized independently
    with ``add_special_tokens=False`` and the resulting IDs are concatenated.
    This makes the returned spans exact for the returned ``input_ids``. Without
    a tokenizer, spans and segment IDs are Unicode-codepoint approximations and
    ``input_ids`` is ``None``; callers must not silently treat those spans as
    model-token spans.
    """
    for name, value in (
        ("voice_prompt", voice_prompt),
        ("style_prompt", style_prompt),
        ("text", text),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    pieces = [
        ("<voice>\n", 0, None),
        (voice_prompt, VOICE_SEGMENT_ID, "voice_span"),
        ("\n</voice>\n\n<style>\n", 0, None),
        (style_prompt, STYLE_SEGMENT_ID, "style_span"),
        ("\n</style>\n\n<text>\n", 0, None),
        (text, TEXT_SEGMENT_ID, "text_span"),
        ("\n</text>", 0, None),
    ]
    full_prompt = "".join(piece[0] for piece in pieces)
    input_ids: list[int] | None = [] if tokenizer is not None else None
    segment_ids: list[int] = []
    spans: dict[str, list[int]] = {}
    for piece_text, segment_id, span_name in pieces:
        piece_ids = (
            encode_text_piece(tokenizer, piece_text)
            if tokenizer is not None
            else list(range(len(piece_text)))
        )
        start = len(segment_ids)
        segment_ids.extend([segment_id] * len(piece_ids))
        if input_ids is not None:
            input_ids.extend(piece_ids)
        if span_name is not None:
            spans[span_name] = [start, len(segment_ids)]

    return {
        "full_prompt": full_prompt,
        **spans,
        "segment_ids": segment_ids,
        "input_ids": input_ids,
        "span_is_exact": tokenizer is not None,
        "tokenization_strategy": (
            "piecewise_no_special_tokens" if tokenizer is not None else "unicode_codepoint_approximation"
        ),
        "span_warning": (
            None
            if tokenizer is not None
            else "No tokenizer was supplied; spans are character-level approximations, not model-token spans."
        ),
    }


__all__ = [
    "VOICE_SEGMENT_ID",
    "STYLE_SEGMENT_ID",
    "TEXT_SEGMENT_ID",
    "build_factorized_prompt",
    "encode_text_piece",
]
