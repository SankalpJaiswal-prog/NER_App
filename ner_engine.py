"""Core NER logic: spaCy baseline + HuggingFace transformer, one common output format.

Every extractor returns a list of dicts: {"text", "label", "start", "end"}
with label in ENTITY_TYPES (PERSON, ORG, LOC).
"""
from __future__ import annotations

from functools import lru_cache
from html import escape

ENTITY_TYPES = ["PERSON", "ORG", "LOC"]

SPACY_MODEL = "en_core_web_sm"
HF_MODEL = "dslim/bert-base-NER"

MAX_CHARS = 20000   # refuse very long input
CHUNK_CHARS = 1000  # transformer input is split into chunks of about this size

# Map each model's own labels onto our three entity types (anything else is ignored).
SPACY_LABEL_MAP = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC"}
HF_LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}

MODELS = {"spacy": "spaCy (fast baseline)", "transformer": "Transformer (BERT)"}


class NERError(Exception):
    """A problem the user can understand (missing model, input too long, ...)."""


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
    except ImportError as e:
        raise NERError("spaCy is not installed. Run: pip install -r requirements.txt") from e
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as e:
        raise NERError(
            f"spaCy model '{SPACY_MODEL}' is not installed. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from e


@lru_cache(maxsize=1)
def _load_transformer():
    try:
        from transformers import pipeline
    except ImportError as e:
        raise NERError("transformers is not installed. Run: pip install -r requirements.txt") from e
    try:
        return pipeline("token-classification", model=HF_MODEL, aggregation_strategy="first")
    except Exception as e:  # download / network / memory problems
        raise NERError(f"Could not load the transformer model '{HF_MODEL}': {e}") from e


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[int, str]]:
    """Split text into (offset, chunk) pieces, cutting at spaces/newlines where possible."""
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        if j < n:
            cut = max(text.rfind(" ", i, j), text.rfind("\n", i, j))
            if cut > i:
                j = cut
        chunks.append((i, text[i:j]))
        i = j
    return chunks


def extract_spacy(text: str) -> list[dict]:
    doc = _load_spacy()(text)
    ents = []
    for e in doc.ents:
        label = SPACY_LABEL_MAP.get(e.label_)
        if label:
            ents.append({"text": e.text, "label": label, "start": e.start_char, "end": e.end_char})
    return ents


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a span outwards until it covers whole words (fixes entities cut mid-word)."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def _merge_overlaps(spans: list[dict]) -> list[dict]:
    """Merge spans with the same label that overlap or touch (e.g. 'Nandan' + 'Nandan Rao')."""
    merged: list[dict] = []
    for s in sorted(spans, key=lambda s: (s["start"], s["end"])):
        last = merged[-1] if merged else None
        if last and last["label"] == s["label"] and s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged


def extract_transformer(text: str) -> list[dict]:
    pipe = _load_transformer()
    spans = []
    for offset, chunk in chunk_text(text):
        for r in pipe(chunk):
            label = HF_LABEL_MAP.get(r["entity_group"])
            if not label:
                continue
            start, end = r["start"] + offset, r["end"] + offset
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            start, end = _snap_to_words(text, start, end)
            spans.append({"label": label, "start": start, "end": end})
    return [
        {"text": text[s["start"]:s["end"]], "label": s["label"], "start": s["start"], "end": s["end"]}
        for s in _merge_overlaps(spans)
    ]


def extract(text: str, model: str = "spacy") -> list[dict]:
    """Main entry point. Returns entities sorted by position."""
    if model not in MODELS:
        raise NERError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}")
    if not text or not text.strip():
        return []
    if len(text) > MAX_CHARS:
        raise NERError(f"Text is too long ({len(text):,} characters). The limit is {MAX_CHARS:,}.")
    ents = extract_spacy(text) if model == "spacy" else extract_transformer(text)
    return sorted(ents, key=lambda e: (e["start"], e["end"]))


def _keys(ents: list[dict]) -> set[tuple[str, str]]:
    return {(" ".join(e["text"].lower().split()), e["label"]) for e in ents}


def compare_models(ents_a: list[dict], ents_b: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Which (entity, type) pairs do both models find, and which does only one find?"""
    a, b = _keys(ents_a), _keys(ents_b)
    return {"both": sorted(a & b), "only_a": sorted(a - b), "only_b": sorted(b - a)}


def highlight_html(text: str, entities: list[dict], colors: dict[str, str]) -> str:
    """Return HTML with each entity highlighted in its type colour (text is HTML-escaped)."""
    parts, pos = [], 0
    for e in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        if e["start"] < pos:  # skip overlaps
            continue
        parts.append(escape(text[pos:e["start"]]))
        color = colors.get(e["label"], "#DDDDDD")
        parts.append(
            f'<mark style="background:{color};padding:2px 6px;border-radius:4px;color:#000;">'
            f'{escape(text[e["start"]:e["end"]])} '
            f'<small style="font-weight:700;font-size:0.7em;">{e["label"]}</small></mark>'
        )
        pos = e["end"]
    parts.append(escape(text[pos:]))
    return '<div style="line-height:2;font-size:1.05rem;white-space:pre-wrap;">' + "".join(parts) + "</div>"
