"""Streamlit front-end for the NER app.  Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from config import APP_SUBTITLE, APP_TITLE, COLORS, LABEL_NAMES
from ner_engine import MAX_CHARS, MODELS, NERError, compare_models, extract, highlight_html
from samples import SAMPLES

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")

MODE_SPACY = "spaCy (fast baseline)"
MODE_TRANSFORMER = "Transformer (BERT)"
MODE_BOTH = "Compare both"
OWN_TEXT = "(write your own)"


def load_sample():
    choice = st.session_state.sample_choice
    if choice != OWN_TEXT:
        st.session_state.text = SAMPLES[choice]


def entities_table(ents):
    rows = [
        {"Entity": e["text"], "Type": LABEL_NAMES[e["label"]], "Start": e["start"], "End": e["end"]}
        for e in ents
    ]
    return pd.DataFrame(rows, columns=["Entity", "Type", "Start", "End"])


def show_result(title, text, ents, key):
    st.subheader(title)
    if not ents:
        st.info("No entities found.")
        return
    st.markdown(highlight_html(text, ents, COLORS), unsafe_allow_html=True)
    cols = st.columns(len(LABEL_NAMES))
    for col, (label, name) in zip(cols, LABEL_NAMES.items()):
        col.metric(name, sum(1 for e in ents if e["label"] == label))
    df = entities_table(ents)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download table as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"entities_{key}.csv", mime="text/csv", key=f"download_{key}",
    )


# ---------- layout ----------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Model", [MODE_SPACY, MODE_TRANSFORMER, MODE_BOTH])
    wanted = st.multiselect(
        "Entity types to show", list(LABEL_NAMES), default=list(LABEL_NAMES),
        format_func=lambda k: LABEL_NAMES[k],
    )
    st.markdown("**Colour key**")
    for label, name in LABEL_NAMES.items():
        st.markdown(
            f'<span style="background:{COLORS[label]};padding:2px 8px;border-radius:4px;color:#000;">{name}</span>',
            unsafe_allow_html=True,
        )

if "text" not in st.session_state:
    st.session_state.text = ""

st.selectbox("Sample text", [OWN_TEXT] + list(SAMPLES), key="sample_choice", on_change=load_sample)
text = st.text_area("Text to analyse", key="text", height=200,
                    placeholder="Paste or type text here, or pick a sample above.")

if st.button("Extract entities", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    elif len(text) > MAX_CHARS:
        st.error(f"Text is too long ({len(text):,} characters). The limit is {MAX_CHARS:,}.")
    else:
        try:
            keep = lambda ents: [e for e in ents if e["label"] in wanted]
            if mode == MODE_BOTH:
                with st.spinner("Running both models..."):
                    a = keep(extract(text, "spacy"))
                    b = keep(extract(text, "transformer"))
                left, right = st.columns(2)
                with left:
                    show_result(MODELS["spacy"], text, a, "spacy")
                with right:
                    show_result(MODELS["transformer"], text, b, "transformer")
                st.subheader("Where the models agree")
                cmp = compare_models(a, b)
                rows = (
                    [{"Entity": t, "Type": LABEL_NAMES[l], "Found by": "Both"} for t, l in cmp["both"]]
                    + [{"Entity": t, "Type": LABEL_NAMES[l], "Found by": "spaCy only"} for t, l in cmp["only_a"]]
                    + [{"Entity": t, "Type": LABEL_NAMES[l], "Found by": "Transformer only"} for t, l in cmp["only_b"]]
                )
                st.dataframe(pd.DataFrame(rows, columns=["Entity", "Type", "Found by"]),
                             width="stretch", hide_index=True)
            else:
                key = "spacy" if mode == MODE_SPACY else "transformer"
                with st.spinner("Running model..."):
                    ents = keep(extract(text, key))
                show_result(MODELS[key], text, ents, key)
        except NERError as err:
            st.error(str(err))
