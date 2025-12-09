# app.py
"""
Feedback-to-Strategy Streamlit App (Prototype)

Features:
- Single feedback input + batch CSV upload
- Zero-shot classification (OpenAI GPT if API key present; rule-based fallback)
- Summarization, sentiment, confidence score, keyword explanations
- Editable tags / corrections (controllability)
- Dashboard: category distribution + sentiment over time
- Weekly Digest generation (agentic behavior) using aggregated items
- Export processed CSV
"""

import os
import math
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import altair as alt

# Optional: OpenAI for LLM-based classification & summarization
try:
    import openai
except Exception:
    openai = None

# --- Configuration ---
st.set_page_config(page_title="Feedback-to-Strategy", layout="wide")
DEFAULT_CATEGORIES = ["bug", "feature_request", "ux_issue", "performance", "question", "praise", "other"]

# --- Utilities & Fallbacks ---


def load_openai_api_key():
    """Try to load OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    return key


OPENAI_KEY = load_openai_api_key()
if openai and OPENAI_KEY:
    openai.api_key = OPENAI_KEY


def safe_call_llm(prompt: str, system: str = None, model: str = "gpt-4o-mini", max_tokens: int = 512) -> str:
    """
    Safely call OpenAI if available. If not, return an explanatory fallback.
    Note: model name is placeholder; adapt to available models in your environment.
    """
    if not openai or not OPENAI_KEY:
        # Fallback: return empty string that the rest of the code can handle
        return ""
    try:
        # Use the ChatCompletions API (example). Adjust to your OpenAI SDK usage.
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        return ""


# --- Simple rule-based classifiers as fallback ---


def simple_keyword_category(text: str, categories: List[str] = DEFAULT_CATEGORIES) -> Tuple[str, float, List[str]]:
    """
    Rule-based keyword matching to assign a category, a confidence score (0-1),
    and contributing keywords as explanation.
    """
    text_l = text.lower()
    keywords_map = {
        "bug": ["error", "bug", "crash", "fails", "exception", "stack trace", "broken"],
        "feature_request": ["would like", "feature", "wish", "please add", "request", "ability to"],
        "ux_issue": ["confusing", "difficult", "ux", "ui", "navigation", "can't find", "hidden"],
        "performance": ["slow", "lag", "delay", "performance", "timeout"],
        "question": ["how do i", "how to", "where is", "can i", "question", "what is"],
        "praise": ["love", "great", "awesome", "thank you", "amazing", "liked"],
        "other": []
    }
    scores = {c: 0 for c in categories}
    hits = {c: [] for c in categories}
    tokens = re.findall(r"[a-zA-Z]{2,}", text_l)
    for c, kws in keywords_map.items():
        for kw in kws:
            if kw in text_l:
                scores[c] += 1
                hits[c].append(kw)
    # Simple heuristic: pick highest score, confidence is normalized
    best = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values())
    if total == 0:
        # fallback: small confidence for 'other'
        return "other", 0.15, []
    conf = min(0.95, 0.3 + best[1] / max(1, total + 1))  # heuristics
    return best[0], round(conf, 2), hits[best[0]]


def simple_sentiment(text: str) -> Tuple[str, float]:
    """
    Very simple sentiment heuristics: counts positive vs negative words.
    Returns label and confidence 0-1.
    """
    pos = ["love", "great", "good", "amazing", "excellent", "awesome", "happy", "like"]
    neg = ["hate", "bad", "terrible", "awful", "slow", "bug", "issue", "crash", "confusing", "frustrating"]
    text_l = text.lower()
    pscore = sum(text_l.count(w) for w in pos)
    nscore = sum(text_l.count(w) for w in neg)
    if pscore == nscore == 0:
        return "neutral", 0.6
    label = "positive" if pscore > nscore else "negative"
    conf = 0.5 + abs(pscore - nscore) / (abs(pscore - nscore) + 2)
    return label, round(min(conf, 0.99), 2)


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """Naive keyword extraction: top frequent alpha tokens excluding stop words."""
    stop = set(["the", "and", "is", "in", "of", "to", "a", "for", "with", "on", "as", "it", "its", "that", "this"])
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq = {}
    for t in tokens:
        if t in stop: continue
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, f in sorted_tokens[:top_k]]


# --- Processing pipeline ---


def classify_and_explain(item_text: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Core pipeline function: returns a dictionary:
    {
        "text": ...,
        "category": ...,
        "confidence": ...,
        "summary": ...,
        "sentiment": ...,
        "sentiment_conf": ...,
        "keywords": [...],
        "explanation": "..."
    }
    """
    result = {"text": item_text, "ts": datetime.utcnow().isoformat()}
    # 1) Try LLM classification if available
    if openai and OPENAI_KEY and use_llm:
        prompt = (
            "Classify the following user feedback into one of: bug, feature_request, ux_issue, "
            "performance, question, praise, other. Return JSON with keys: category, confidence (0-1), "
            "keywords (list), summary (1-2 sentences), sentiment (positive/neutral/negative), sentiment_confidence.\n\n"
            f"Feedback: '''{item_text}'''"
        )
        llm_out = safe_call_llm(prompt, system="You are a disciplined JSON-output assistant.", max_tokens=512)
        try:
            parsed = json.loads(llm_out)
            # guard for keys
            result.update({
                "category": parsed.get("category", "other"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "keywords": parsed.get("keywords", []) or extract_keywords(item_text),
                "summary": parsed.get("summary", "") or item_text[:200],
                "sentiment": parsed.get("sentiment", "neutral"),
                "sentiment_conf": float(parsed.get("sentiment_confidence", 0.5)),
                "explanation": parsed.get("explanation", "")
            })
            return result
        except Exception:
            # fall through to rule-based if parsing fails
            pass

    # 2) Fallback rule-based pipeline
    cat, conf, contributing = simple_keyword_category(item_text)
    sentiment, sconf = simple_sentiment(item_text)
    keywords = extract_keywords(item_text)
    # short summary: first 1-2 lines or truncated text
    summary = item_text.strip().replace("\n", " ")
    if len(summary) > 200:
        # try sentence-based truncation
        sentences = re.split(r'(?<=[.!?]) +', summary)
        summary = sentences[0] if sentences else summary[:200]
    explanation = f"Matched keywords: {', '.join(contributing)}" if contributing else f"Top words: {', '.join(keywords[:3])}"
    result.update({
        "category": cat,
        "confidence": conf,
        "keywords": keywords,
        "summary": summary,
        "sentiment": sentiment,
        "sentiment_conf": sconf,
        "explanation": explanation
    })
    return result


# --- Streamlit UI ---


def sidebar_controls():
    st.sidebar.header("Feedback-to-Strategy — Controls")
    use_llm = st.sidebar.checkbox("Use LLM for classification/summaries (requires OPENAI_API_KEY)", value=bool(openai and OPENAI_KEY))
    show_explanations = st.sidebar.checkbox("Show explanations/highlights", value=True)
    return use_llm, show_explanations


def render_input_panel(use_llm: bool):
    st.header("Input Feedback")
    col1, col2 = st.columns([2, 1])
    with col1:
        text = st.text_area("Paste a feedback item (single message) or type one:", height=140)
        if st.button("Analyze single feedback"):
            if not text.strip():
                st.warning("Please paste or type some feedback first.")
            else:
                item = classify_and_explain(text, use_llm=use_llm)
                st.session_state.processed = st.session_state.processed.append(item, ignore_index=True)
                st.success("Processed and added to session data.")
    with col2:
        st.markdown("**Batch upload**")
        uploaded = st.file_uploader("Upload CSV of feedback (column named 'text' recommended)", type=["csv"])
        if uploaded:
            df_batch = pd.read_csv(uploaded)
            if "text" not in df_batch.columns:
                st.warning("CSV missing `text` column. Please upload a CSV with a 'text' column.")
            else:
                if st.button("Process CSV"):
                    processed_rows = []
                    for idx, row in df_batch.iterrows():
                        item = classify_and_explain(str(row["text"]), use_llm=use_llm)
                        # carry over some metadata if present
                        for k in ("id", "user", "date"):
                            if k in row:
                                item[k] = row[k]
                        processed_rows.append(item)
                    st.session_state.processed = pd.concat([st.session_state.processed, pd.DataFrame(processed_rows)], ignore_index=True)
                    st.success(f"Processed {len(processed_rows)} rows.")
    st.markdown("---")


def render_insights_table(show_explanations: bool):
    st.subheader("Insights (Processed Feedback)")
    if st.session_state.processed.empty:
        st.info("No processed feedback yet. Add single feedback or upload a CSV to get started.")
        return

    df = st.session_state.processed.copy()
    # display a simplified table with edit controls
    display_cols = ["text", "category", "confidence", "summary", "sentiment", "sentiment_conf", "keywords"]
    st.dataframe(df[display_cols].rename(columns={"sentiment_conf": "sent_conf"}).head(200))

    # Let user select a row to edit
    idx = st.number_input("Select row index to edit / inspect", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    row = df.iloc[int(idx)].to_dict()
    st.markdown("**Selected item**")
    st.write(row.get("text", ""))

    # Editable fields
    new_category = st.selectbox("Category", DEFAULT_CATEGORIES, index=DEFAULT_CATEGORIES.index(row.get("category")) if row.get("category") in DEFAULT_CATEGORIES else 0)
    new_conf = st.slider("Confidence", 0.0, 1.0, float(row.get("confidence", 0.5)))
    new_summary = st.text_area("Summary", value=row.get("summary", ""))
    new_sentiment = st.selectbox("Sentiment", ["positive", "neutral", "negative"], index=["positive", "neutral", "negative"].index(row.get("sentiment", "neutral")))
    # Save edits
    if st.button("Save edits to selected row"):
        st.session_state.processed.at[int(idx), "category"] = new_category
        st.session_state.processed.at[int(idx), "confidence"] = float(new_conf)
        st.session_state.processed.at[int(idx), "summary"] = new_summary
        st.session_state.processed.at[int(idx), "sentiment"] = new_sentiment
        st.success("Saved edits.")

    # Explanations
    if show_explanations:
        st.markdown("**Explanation / Keywords**")
        st.write(row.get("explanation", ""))
        st.write("Keywords:", ", ".join(row.get("keywords", [])))
    st.markdown("---")


def render_dashboard():
    st.subheader("Dashboard")
    if st.session_state.processed.empty:
        st.info("No data yet — dashboard will populate after processing feedback.")
        return
    df = st.session_state.processed.copy()
    # Normalize fields
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df["ts"] = pd.Timestamp.now()
    # Category distribution
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    chart = alt.Chart(counts).mark_bar().encode(x=alt.X("category:N", sort="-y"), y="count:Q", color="category:N").properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    # Sentiment over time (daily avg)
    df["date"] = df["ts"].dt.date
    sent_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sent_score"] = df["sentiment"].map(sent_map).fillna(0)
    daily = df.groupby("date")["sent_score"].mean().reset_index()
    line = alt.Chart(daily).mark_line(point=True).encode(x="date:T", y="sent_score:Q").properties(height=200)
    st.altair_chart(line, use_container_width=True)

    # Emerging keywords
    all_keywords = df["keywords"].explode().dropna().astype(str)
    top = all_keywords.value_counts().reset_index().rename(columns={"index": "kw", 0: "count"}).head(15)
    if not top.empty:
        wc = alt.Chart(top).mark_bar().encode(y=alt.Y("kw:N", sort="-x"), x="count:Q").properties(height=300)
        st.altair_chart(wc, use_container_width=True)
    st.markdown("---")


def generate_weekly_digest(use_llm: bool) -> str:
    """
    Agentic weekly digest: aggregates last 7 days and produces a strategic summary.
    If LLM available, use it to create more polished summary; else, build a rule-based digest.
    """
    if st.session_state.processed.empty:
        return "No data available for digest."

    df = st.session_state.processed.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    week = df[df["ts"] >= cutoff]
    if week.empty:
        return "No feedback in the last 7 days."

    # Build aggregated facts
    top_categories = week["category"].value_counts().head(5).to_dict()
    pos = (week["sentiment"] == "positive").sum()
    neg = (week["sentiment"] == "negative").sum()
    total = len(week)
    keywords = week["keywords"].explode().dropna().value_counts().head(10).to_dict()
    sample_issues = week.sort_values("confidence").head(5)["text"].tolist()

    if openai and OPENAI_KEY and use_llm:
        prompt = (
            "You are an insights analyst. Create a concise executive weekly digest from the aggregated feedback facts below.\n\n"
            f"Counts by category: {json.dumps(top_categories)}\n"
            f"Positive: {pos}, Negative: {neg}, Total items: {total}\n"
            f"Top keywords: {json.dumps(keywords)}\n"
            f"Representative low-confidence items: {json.dumps(sample_issues)}\n\n"
            "Produce:\n- 3-sentence executive summary\n- Top 3 recommended actions (1 line each)\n- One short quote from representative feedback (use an item from sample_issues)\n"
        )
        llm_out = safe_call_llm(prompt, system="You produce short, clear, executive summaries.", max_tokens=300)
        return llm_out or "LLM returned empty — fallback digest below.\n\n" + simple_digest_text(top_categories, pos, neg, total, keywords, sample_issues)
    else:
        return simple_digest_text(top_categories, pos, neg, total, keywords, sample_issues)


def simple_digest_text(top_categories, pos, neg, total, keywords, sample_issues):
    lines = []
    lines.append(f"In the last 7 days we processed {total} feedback items. Top categories: " + ", ".join([f"{k}({v})" for k, v in top_categories.items()]))
    lines.append(f"Sentiment: {pos} positive / {neg} negative.")
    top_k = ", ".join(list(keywords.keys())[:5]) if keywords else "none"
    lines.append(f"Top keywords: {top_k}.")
    lines.append("\nRecommended actions:")
    # choose recommendations heuristically
    recs = []
    if "bug" in top_categories:
        recs.append("Prioritize top reported bugs into next sprint; assign owner and ETA.")
    if "ux_issue" in top_categories:
        recs.append("Schedule a quick UX review for the top UX complaints and prototype fixes.")
    if not recs:
        recs.append("Investigate top categories and add high-confidence items to backlog grooming.")
    lines.extend([f"- {r}" for r in recs[:3]])
    lines.append("\nRepresentative quote:")
    if sample_issues:
        lines.append(f"\"{sample_issues[0][:200]}\"")
    return "\n".join(lines)


def export_processed_csv():
    if st.session_state.processed.empty:
        st.info("No data to export.")
        return
    csv = st.session_state.processed.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", data=csv, file_name="processed_feedback.csv", mime="text/csv")


# --- Initialize session state ---
if "processed" not in st.session_state:
    st.session_state.processed = pd.DataFrame(columns=[
        "text", "category", "confidence", "summary", "sentiment", "sentiment_conf", "keywords", "explanation", "ts"
    ])

# --- App layout ---
st.title("Feedback-to-Strategy — Prototype")
st.markdown("Turn raw feedback into actionable insights. Built with HAI principles: transparency, controllability, and explainability.")

use_llm, show_explanations = sidebar_controls()

# Main panels
left, right = st.columns([2, 1])
with left:
    render_input_panel(use_llm)
    render_insights_table(show_explanations)

with right:
    render_dashboard()
    st.subheader("Agent Actions")
    if st.button("Generate Weekly Digest"):
        digest = generate_weekly_digest(use_llm=use_llm)
        st.text_area("Weekly Digest (editable)", value=digest, height=260)
        st.success("Digest generated. Edit and share via copy or export.")
    st.markdown("### Export / Utilities")
    export_processed_csv()
    st.markdown("### Tips")
    st.write("- Use the sidebar to toggle LLM usage.")
    st.write("- Edit any misclassified item before exporting or acting on insights.")
    st.write("- The weekly digest is agentic: it aggregates and recommends next steps.")

st.markdown("---")
st.caption("Prototype — replace safe_call_llm with your production LLM integration and add authentication for multi-user deployment.")
