"""
app.py  -  News Summarization and Key Information Extraction Using NLP
Run with:  streamlit run app.py
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import evaluate
from src.keyword_extractor import extract_keywords
from src.ner import GROUPS, count_entities, extract_entities, load_spacy_model
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import clean_text, show_pipeline_steps, split_headline, split_sentences
from src.summarizer import summarize
from src.utils import FriendlyError, ensure_nltk_resources, read_txt_file, validate_text, word_count

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Technology - AI in rural healthcare": "sample_technology.txt",
    "Sports - Cricket final": "sample_sports.txt",
    "Politics / Business - Small business fund": "sample_politics_business.txt",
}
GROUP_ICONS = {"Person": "👤", "Organization": "🏢", "Location": "📍",
               "Date": "📅", "Money": "💰", "Event": "🎉"}
GROUP_COLORS = {"Person": "#dbeafe", "Organization": "#dcfce7", "Location": "#fee2e2",
                "Date": "#fef9c3", "Money": "#ede9fe", "Event": "#ffedd5"}

st.set_page_config(page_title="News Summarization using NLP", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .banner {background: linear-gradient(90deg, #1e3a8a, #2563eb); padding: 22px 28px;
             border-radius: 12px; margin-bottom: 18px;}
    .banner h1 {color: white; font-size: 1.7rem; margin: 0; padding: 0;}
    .banner p {color: #dbeafe; margin: 6px 0 0 0;}
    .chip {display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
           border-radius: 16px; font-size: 0.9rem; color: #111827; background: #e0e7ff;}
    .summary-box {background: rgba(37, 99, 235, 0.08); border-left: 5px solid #2563eb;
                  padding: 14px 18px; border-radius: 6px; line-height: 1.7;}
    </style>
    <div class="banner">
        <h1>📰 News Summarization and Key Information Extraction Using NLP</h1>
        <p>Extractive summary &bull; Keywords &bull; Named entities &bull; Article statistics</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading spaCy model...")
def get_nlp():
    return load_spacy_model()


def chips(items, color="#e0e7ff"):
    """Return HTML 'chips' for a list of strings (text is escaped for safety)."""
    return "".join(
        f'<span class="chip" style="background:{color}">{html.escape(str(i))}</span>' for i in items
    )


def run_analysis(raw_text, length):
    """Run the complete NLP pipeline and return everything the dashboard needs."""
    ensure_nltk_resources()
    headline, body = split_headline(raw_text)
    text = clean_text(body)
    validate_text(text)

    result = summarize(text, length)
    keywords = extract_keywords(text)

    entities, ner_error = None, None
    try:
        entities = extract_entities(text, get_nlp())
    except FriendlyError as err:           # the app still works without NER
        ner_error = str(err)

    summary_words = word_count(result["summary"])
    original_words = word_count(text)
    stats = {
        "Original Words": original_words,
        "Summary Words": summary_words,
        "Original Sentences": len(result["sentences"]),
        "Summary Sentences": len(result["selected"]),
        "Summary Percentage": f"{summary_words / original_words * 100:.1f}%",
        "Compression Ratio": f"{(1 - summary_words / original_words) * 100:.1f}%",
        "Keywords": len(keywords),
        "Entities": count_entities(entities) if entities else 0,
    }
    return {"headline": headline, "text": text, "length": length, "result": result,
            "keywords": keywords, "entities": entities, "ner_error": ner_error, "stats": stats}


def get_input_text(method):
    """Return (text, error_message) according to the chosen input method."""
    if method == "✍️ Paste text":
        return st.session_state.get("pasted_text", ""), None
    if method == "📄 Upload TXT":
        file = st.session_state.get("txt_file")
        if file is None:
            return "", "Please upload a .txt file."
        return read_txt_file(file), None
    if method == "📕 Upload PDF":
        file = st.session_state.get("pdf_file")
        if file is None:
            return "", "Please upload a .pdf file."
        return extract_text_from_pdf(file), None
    name = st.session_state.get("sample_name")
    return (DATA_DIR / SAMPLES[name]).read_text(encoding="utf-8"), None


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("📥 Input")
    method = st.radio(
        "Choose how to give the article",
        ["✍️ Paste text", "📄 Upload TXT", "📕 Upload PDF", "🧪 Use a sample article"],
        horizontal=True,
    )

    if method == "✍️ Paste text":
        st.text_area("Paste the news article here", height=260, key="pasted_text",
                     placeholder="Paste a long news article (at least 50 words)...")
    elif method == "📄 Upload TXT":
        st.file_uploader("Upload a .txt file", key="txt_file")
    elif method == "📕 Upload PDF":
        st.file_uploader("Upload a .pdf file", key="pdf_file")
    else:
        st.selectbox("Select a sample", list(SAMPLES), key="sample_name")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        length = st.radio("Summary length", ["Short", "Medium", "Long"], index=1, horizontal=True,
                          help="Short = 20% of sentences, Medium = 30%, Long = 40%")
    with col_b:
        st.write("")
        analyze = st.button("🔍 Analyze Article", type="primary")

if analyze:
    st.session_state.pop("analysis", None)
    try:
        raw_text, message = get_input_text(method)
        if message:
            st.warning(message)
        else:
            with st.spinner("Analysing the article..."):
                st.session_state["analysis"] = run_analysis(raw_text, length)
    except FriendlyError as err:
        st.error(f"⚠️ {err}")
    except Exception:
        st.error("⚠️ Something went wrong while analysing the article. "
                 "Please check the input and try again.")

# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------
analysis = st.session_state.get("analysis")

if analysis:
    result, stats = analysis["result"], analysis["stats"]

    # 1. Summary ------------------------------------------------------------
    with st.container(border=True):
        st.subheader(f"📝 Generated Summary ({analysis['length']})")
        if analysis["headline"]:
            st.caption(f"Headline: {analysis['headline']}")
        st.markdown(f'<div class="summary-box">{html.escape(result["summary"])}</div>',
                    unsafe_allow_html=True)
        st.download_button("⬇️ Download summary", result["summary"], file_name="summary.txt")

    # 2. Keywords -----------------------------------------------------------
    with st.container(border=True):
        st.subheader("🔑 Important Keywords")
        if analysis["keywords"]:
            st.markdown(chips([k for k, _ in analysis["keywords"]]), unsafe_allow_html=True)
            df_kw = pd.DataFrame(analysis["keywords"], columns=["Keyword", "TF-IDF score"])
            st.bar_chart(df_kw.set_index("Keyword"))
        else:
            st.info("No keywords could be extracted from this article.")

    # 3. Entities -----------------------------------------------------------
    with st.container(border=True):
        st.subheader("👤 Named Entities")
        if analysis["entities"] is None:
            st.warning(f"⚠️ {analysis['ner_error']}")
        else:
            for row in (GROUPS[:3], GROUPS[3:]):
                columns = st.columns(3)
                for col, group in zip(columns, row):
                    with col:
                        st.markdown(f"**{GROUP_ICONS[group]} {group.upper()}**")
                        names = [name for name, _ in analysis["entities"][group]]
                        if names:
                            st.markdown(chips(names, GROUP_COLORS[group]), unsafe_allow_html=True)
                        else:
                            st.caption("None found")

    # 4. Statistics ---------------------------------------------------------
    with st.container(border=True):
        st.subheader("📊 Article Statistics")
        items = list(stats.items())
        for start in (0, 4):
            columns = st.columns(4)
            for col, (label, value) in zip(columns, items[start:start + 4]):
                col.metric(label, value)

    # Explainability --------------------------------------------------------
    with st.container(border=True):
        st.subheader("🔬 How it worked (for explanation)")

        with st.expander("Preprocessing steps on one sentence"):
            sample_sentence = next((s for s in result["sentences"] if len(s.split()) >= 8),
                                   result["sentences"][0])
            for step, output in show_pipeline_steps(sample_sentence).items():
                st.markdown(f"**{step}:**")
                st.code(output if isinstance(output, str) else ", ".join(output), language=None)

        with st.expander("Sentence scores and ranking"):
            order = sorted(range(len(result["scores"])), key=lambda i: result["scores"][i], reverse=True)
            rank = {idx: pos + 1 for pos, idx in enumerate(order)}
            df_scores = pd.DataFrame({
                "No.": [i + 1 for i in range(len(result["sentences"]))],
                "Sentence": [s if len(s) < 110 else s[:107] + "..." for s in result["sentences"]],
                "Score": [round(x, 3) for x in result["scores"]],
                "Rank": [rank[i] for i in range(len(result["sentences"]))],
                "In summary": ["✅" if i in result["selected"] else "" for i in range(len(result["sentences"]))],
            })
            st.dataframe(df_scores, hide_index=True)

        with st.expander("Optional: evaluate with a reference summary (ROUGE)"):
            reference = st.text_area("Paste a human-written reference summary", height=120, key="reference")
            if reference.strip():
                rouge = evaluate(reference, result["summary"])
                df_rouge = pd.DataFrame(rouge).T.rename(columns=str.capitalize)
                st.dataframe(df_rouge)
else:
    st.info("Choose an input method, then click **Analyze Article**. "
            "You can try the built-in sample articles first.")

st.caption("B.Tech AI & DS NLP Mini Project - Python, NLTK, spaCy, Scikit-learn, Streamlit")
