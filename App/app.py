"""EntityLens - Named Entity Recognition for News Articles.

Run:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    streamlit run app.py

Backend: spaCy (en_core_web_sm) by default. Set BACKEND = "bert" to use
dslim/bert-base-NER from Hugging Face instead (needs transformers + torch).
"""

import html

import streamlit as st

st.set_page_config(page_title="EntityLens", page_icon="🔎", layout="wide")

BACKEND = "spacy"  # "spacy" or "bert"

SAMPLE = (
    "Apple opened a new retail store in Bengaluru on Saturday. "
    "Store manager Priya Nair said the opening had been planned for many months."
)

# Entity colours (from the design spec) and text colour.
COLORS = {"PER": "#DCEBFF", "ORG": "#D8F3E5", "LOC": "#FCE7D6"}
TEXT_COLOR = "#1E2A3A"

LABEL_NAMES = {"PER": "Person", "ORG": "Organization", "LOC": "Location"}

SPACY_MAP = {"PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC"}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading language model...")
def load_model():
    if BACKEND == "bert":
        from transformers import pipeline

        return pipeline(
            "ner", model="dslim/bert-base-NER", aggregation_strategy="simple"
        )
    import spacy

    return spacy.load("en_core_web_sm")


def extract_entities(text):
    """Return a list of {text, label, start, end} with end excluded, sorted by start."""
    model = load_model()
    found = []
    if BACKEND == "bert":
        for r in model(text):
            label = r["entity_group"]
            if label in COLORS:
                found.append(
                    {
                        "text": text[r["start"] : r["end"]],
                        "label": label,
                        "start": int(r["start"]),
                        "end": int(r["end"]),
                    }
                )
    else:
        for ent in model(text).ents:
            label = SPACY_MAP.get(ent.label_)
            if label:
                found.append(
                    {
                        "text": ent.text,
                        "label": label,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )
    found.sort(key=lambda e: (e["start"], -e["end"]))
    clean, last_end = [], 0
    for e in found:  # drop overlaps so highlighting stays valid
        if e["start"] >= last_end:
            clean.append(e)
            last_end = e["end"]
    return clean


# --------------------------------------------------------------------------
# State and callbacks
# --------------------------------------------------------------------------
def run_extraction():
    text = st.session_state.text
    if not text.strip():
        st.session_state.results = None
        st.session_state.error = "Add a news paragraph first, then choose Extract Entities."
        return
    try:
        entities = extract_entities(text)
    except (OSError, ImportError) as exc:
        st.session_state.results = None
        st.session_state.error = (
            "The language model could not be loaded. Run: "
            "python -m spacy download en_core_web_sm  "
            f"({type(exc).__name__})"
        )
        return
    st.session_state.error = None
    st.session_state.results = {"text": text, "entities": entities}


def load_sample():
    st.session_state.text = SAMPLE
    run_extraction()


if "text" not in st.session_state:
    st.session_state.text = SAMPLE
    st.session_state.error = None
    st.session_state.results = None
    run_extraction()


# --------------------------------------------------------------------------
# Page + styles
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #1E2A3A;
  --muted: #6B7787;
  --line: #E3E8EF;
  --panel: #F5F7FB;
  --navy: #1F2F4D;
  --action: #254B7E;
  --per: #DCEBFF; --org: #D8F3E5; --loc: #FCE7D6;
}

html, body, [class*="st-"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
html, body, .stApp { color-scheme: light; }
.stApp { background: #FFFFFF; color: var(--ink); }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 1440px; padding: 0 44px 24px 44px !important; }

/* top bar */
.topbar { display:flex; align-items:center; justify-content:space-between;
  padding: 16px 0; border-bottom: 1px solid var(--line); margin: 0 -44px 40px -44px;
  padding-left: 44px; padding-right: 44px; }
.brand { display:flex; align-items:center; gap:14px; }
.logo { width:46px; height:46px; border-radius:12px; background:var(--navy);
  display:flex; align-items:center; justify-content:center; }
.brand-name { font-size:20px; font-weight:600; line-height:1.2; color:var(--ink); }
.brand-sub { font-size:12px; color:var(--muted); }
.pill { font-size:12px; color:var(--muted); border:1px solid var(--line);
  border-radius:999px; padding:7px 16px; }

.headline { font-size:26px; font-weight:500; color:var(--ink); margin:0 0 6px 0; letter-spacing:-0.01em; }
.subhead { font-size:13px; color:var(--muted); margin:0 0 28px 0; }

/* cards */
.st-key-input_card, .card {
  background:#fff; border:1px solid var(--line); border-radius:16px; padding:22px;
}
.card { margin-bottom:20px; }
.card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.card-title { display:flex; align-items:center; gap:10px; font-size:16px; font-weight:500; color:var(--ink); }
.step { font-size:11px; color:var(--muted); background:#EEF2F8; border-radius:6px; padding:3px 7px; }
.meta { font-size:11px; color:var(--muted); }
.hint { font-size:12px; color:var(--muted); margin:0; }

/* text area */
.st-key-input_card div[data-baseweb="textarea"] {
  border:1px solid #BCC7D6; border-radius:12px; background:#fff; }
.st-key-input_card div[data-baseweb="textarea"]:focus-within {
  border-color: var(--action); box-shadow: 0 0 0 3px rgba(37,75,126,.18); }
.st-key-input_card div[data-baseweb="base-input"],
.st-key-input_card div[data-baseweb="textarea"] > div { background:#FFFFFF !important; border-radius:12px; }
.st-key-input_card textarea { font-size:14px; line-height:1.9; padding:14px 16px;
  background:#FFFFFF !important; color:#1E2A3A !important; -webkit-text-fill-color:#1E2A3A !important;
  caret-color:#1E2A3A; }
.st-key-input_card textarea::placeholder { color:#8A95A5 !important; -webkit-text-fill-color:#8A95A5 !important; }
.counter { font-size:11px; color:var(--muted); margin:-4px 0 14px 4px; }

/* buttons */
.st-key-input_card .stButton { width:100%; }
.st-key-input_card .stButton button { width:100%; height:44px; border-radius:10px;
  font-size:14px; font-weight:500; font-family:'Inter',sans-serif; }
.st-key-input_card button[kind="primary"],
.st-key-input_card button[data-testid="stBaseButton-primary"] {
  background:var(--action); border:1px solid var(--action); color:#fff; }
.st-key-input_card button[kind="primary"]:hover,
.st-key-input_card button[data-testid="stBaseButton-primary"]:hover { background:#1D3F6A; border-color:#1D3F6A; color:#fff; }
.st-key-input_card button[kind="secondary"],
.st-key-input_card button[data-testid="stBaseButton-secondary"] {
  background:#fff; border:1px solid var(--line); color:var(--ink); }
.st-key-input_card button[kind="secondary"]:hover,
.st-key-input_card button[data-testid="stBaseButton-secondary"]:hover { border-color:var(--action); color:var(--action); }
.st-key-input_card button:focus-visible { outline:3px solid rgba(37,75,126,.35); outline-offset:2px; }

/* highlighted paragraph */
.para { background:var(--panel); border-radius:12px; padding:20px 20px; margin-top:14px;
  font-size:14.5px; line-height:2.2; color:var(--ink); white-space:pre-wrap; }
.ent { color:#1E2A3A; font-weight:500; border-radius:6px; padding:3px 8px; }
.ent-PER, .chip-PER { background:#DCEBFF; }
.ent-ORG, .chip-ORG { background:#D8F3E5; }
.ent-LOC, .chip-LOC { background:#FCE7D6; }

.legend { display:flex; align-items:center; gap:10px; margin-top:14px; font-size:11px; color:var(--muted); flex-wrap:wrap; }
.chip { color:#1E2A3A; font-size:11px; font-weight:500; border-radius:6px; padding:4px 10px; }

/* table */
.ent-table { width:100%; border-collapse:collapse; margin-top:16px; font-size:14px; }
.ent-table th { text-align:left; font-size:12px; font-weight:400; color:var(--muted);
  background:var(--panel); padding:12px 12px; }
.ent-table td { padding:14px 12px; border-bottom:1px solid var(--line); color:var(--ink); }
.ent-table th:nth-child(1), .ent-table td:nth-child(1) { width:34%; }
.ent-table th:nth-child(2), .ent-table td:nth-child(2) { width:28%; }
.badge { color:#1E2A3A; font-size:10.5px; font-weight:600; border-radius:6px; padding:4px 9px; display:inline-block; }
.table-note { font-size:11px; color:var(--muted); margin:14px 0 0 0; }

.empty { font-size:13px; color:var(--muted); padding:26px 4px 8px 4px; }
.notice { font-size:13px; color:#8A4B12; background:#FCE7D6; border-radius:10px; padding:12px 14px; margin-top:12px; color:#1E2A3A; }

.footer { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:90px; }

@media (max-width: 800px) {
  .block-container { padding: 0 16px 24px 16px !important; }
  .topbar { margin: 0 -16px 24px -16px; padding-left:16px; padding-right:16px; }
  .footer { flex-direction:column; gap:6px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2'
    'M8 20H6a2 2 0 0 1-2-2v-2M8 12h8"/></svg>'
)

st.markdown(
    '<div class="topbar"><div class="brand"><div class="logo">' + LOGO_SVG + "</div>"
    '<div><div class="brand-name">EntityLens</div>'
    '<div class="brand-sub">Named Entity Recognition for News Articles</div></div></div>'
    '<div class="pill">NLP demonstration</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="headline">Discover the people, places, and organizations in the news.</h1>'
    '<p class="subhead">Add a news paragraph, extract its entities, and inspect the results '
    "alongside the original text.</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def plural(n, one, many):
    return f"{n} {one if n == 1 else many}"


def highlighted_html(text, ents):
    out, pos = [], 0
    for e in ents:
        out.append(html.escape(text[pos : e["start"]]))
        out.append(f'<span class="ent ent-{e["label"]}">{html.escape(e["text"])}</span>')
        pos = e["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


def legend_html():
    chips = "".join(
        f'<span class="chip chip-{k}">{LABEL_NAMES[k]} ({k})</span>' for k in COLORS
    )
    return f'<div class="legend"><span>Legend</span>{chips}</div>'


def card_head(step, title, meta=""):
    meta_html = f'<span class="meta">{meta}</span>' if meta else ""
    step_html = f'<span class="step">{step}</span>' if step else ""
    return (
        f'<div class="card-head"><div class="card-title">{step_html}'
        f"{title}</div>{meta_html}</div>"
    )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
left, right = st.columns([5, 8], gap="large")

with left:
    with st.container(key="input_card"):
        st.markdown(
            card_head("01", "News paragraph")
            + '<p class="hint" style="margin-bottom:14px">Paste your article text below, '
            "or start with the sample.</p>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "News paragraph",
            key="text",
            height=190,
            label_visibility="collapsed",
            placeholder="Paste a news paragraph here...",
        )
        txt = st.session_state.text
        st.markdown(
            f'<div class="counter">{len(txt)} characters · {len(txt.split())} words</div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns([2, 1], gap="small")
        with b1:
            st.button("Extract Entities", type="primary", on_click=run_extraction)
        with b2:
            st.button("Load Sample", on_click=load_sample)
        st.markdown(
            '<p class="hint" style="margin-top:12px">Plain text works best. '
            "Punctuation is preserved.</p>",
            unsafe_allow_html=True,
        )

with right:
    res = st.session_state.results
    err = st.session_state.error

    if res is None:
        message = err or "Choose Extract Entities to see the highlighted paragraph and the entity table."
        st.markdown(
            '<div class="card">' + card_head("02", "Highlighted paragraph")
            + f'<div class="empty">{html.escape(message)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card">' + card_head("", "Extracted entities")
            + '<div class="empty">No results yet.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        ents = res["entities"]
        n = len(ents)

        para = highlighted_html(res["text"], ents)
        note = ""
        if n == 0:
            note = (
                '<div class="notice">No people, organizations, or locations were found. '
                "Try a longer news paragraph with named subjects.</div>"
            )
        st.markdown(
            '<div class="card">'
            + card_head("02", "Highlighted paragraph", plural(n, "entity", "entities") + " detected")
            + f'<div class="para">{para}</div>'
            + legend_html()
            + note
            + "</div>",
            unsafe_allow_html=True,
        )

        rows = "".join(
            f"<tr><td>{html.escape(e['text'])}</td>"
            f'<td><span class="badge chip-{e["label"]}">{e["label"]}</span></td>'
            f"<td>[{e['start']}, {e['end']})</td></tr>"
            for e in ents
        )
        table = (
            '<table class="ent-table"><thead><tr><th>Entity</th><th>Type</th><th>Position</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
            if n
            else '<div class="empty">No entities to list.</div>'
        )
        st.markdown(
            '<div class="card">'
            '<div class="card-head"><div class="card-title">Extracted entities</div>'
            f'<span class="meta">{plural(n, "result", "results")}</span></div>'
            + table
            + '<p class="table-note">Positions are 0-based character offsets: [start, end), '
            "with the end excluded. Spaces count.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer"><span>EntityLens · University NLP project</span>'
    "<span>English text · Person, Organization, Location</span></div>",
    unsafe_allow_html=True,
)