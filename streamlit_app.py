import streamlit as st
from streamlit import caching
from spacy_streamlit import load_model, visualize_tokens, visualize_parser
from scispacy.custom_sentence_segmenter import pysbd_sentencizer
import spacy
import time
import pandas as pd

st.set_page_config(
    page_title="Scispacy-Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------- side bar --------------------------------- #

spacy_model = st.sidebar.selectbox(
    "Model name", ["en_core_sci_sm", "en_core_sci_md", "en_core_sci_lg"]
)
nlp = load_model(spacy_model)
st.sidebar.subheader("Pipeline info")
desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
st.sidebar.markdown(desc, unsafe_allow_html=True)
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""
st.sidebar.markdown(
    FOOTER,
    unsafe_allow_html=True,
)

# ------------------------------- main contents ------------------------------ #


@st.cache()
def get_doc_info(doc):
    doc_info = []
    for token in doc:
        token_info = [
            token.text,
            f"[{token.pos_}]: " + str(spacy.explain(token.pos_)),
            f"[{token.tag_}]: " + str(spacy.explain(token.tag_)),
            f"[{token.dep_}]: " + str(spacy.explain(token.dep_)),
        ]
        doc_info.append(token_info)
    return doc_info


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def process_text(nlp, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    return nlp(text)


if "pysbd_sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("pysbd_sentencizer", before="tok2vec")
with st.form(key="my_form"):
    text = st.text_area(
        label="Text to analyze",
        value="This package contains utilities for visualizing spaCy models and building interactive spaCy-powered apps with Streamlit.",
    )
    submit_button = st.form_submit_button(label="visualize")
    start = time.time()
    doc = nlp(text)
    elapsed_time = time.time() - start
    st.write("processing time:\t", elapsed_time)

# 依存関係を表示する
visualize_parser(doc, title="")

# 主なtokenの情報を表示する
df = pd.DataFrame(get_doc_info(doc), columns=["text", "pos", "tag", "dep"])
df.set_index("text", inplace=True)
st.table(df)

# 全てのtokenの情報を表示する
visualize_tokens(
    doc,
    attrs=[
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "head",
        "morph",
        "ent_type_",
        "ent_iob_",
        "shape_",
        "is_alpha",
        "is_ascii",
        "is_digit",
        "is_punct",
        "like_num",
        "is_sent_start",
    ],
    title="All Token attributes",
)
