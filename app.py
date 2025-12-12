import streamlit as st
import pandas as pd
import os
import json
import google.generativeai as genai

# -------------------------------------------------------------
# 1. CONFIGURE GEMINI
# -------------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "models/gemini-2.0-flash"   # or "gemini-1.5-pro"

model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------------------
# 2. GEMINI FUNCTION: CLASSIFY + SUMMARIZE + SENTIMENT
# -------------------------------------------------------------
def analyze_feedback(text):
    prompt = f"""
You are an AI feedback insights analyst.

Classify the feedback into the following:
- category: one of [bug, feature_request, ux_issue, performance, documentation, other]
- sentiment: positive, neutral, or negative
- summary: 1–2 sentence explanation of the key issue
- key_phrases: 3–6 important phrases or signals
- explanation: Brief reasoning behind the classification

Return JSON only.

Feedback: \"{text}\"
    """

    response = model.generate_content(prompt)
    raw = response.text

    # Parse JSON safely
    try:
        data = json.loads(raw)
    except:
        data = {
            "category": "other",
            "sentiment": "neutral",
            "summary": raw,
            "key_phrases": [],
            "explanation": "Model output was not valid JSON."
        }

    # -------------------------------------------------------------
    # RULE-BASED CONFIDENCE SCORE
    # -------------------------------------------------------------
    confidence = 0

    # weight 1 — number of key phrases (max 40%)
    kp = len(data.get("key_phrases", []))
    confidence += min(kp / 6, 1) * 0.4

    # weight 2 — sentiment clarity (40%)
    if data.get("sentiment") in ["positive", "negative"]:
        confidence += 0.4
    else:
        confidence += 0.2

    # weight 3 — category present (20%)
    if data.get("category") not in ["other", None, ""]:
        confidence += 0.2

    data["confidence"] = round(confidence, 2)

    return data

# -------------------------------------------------------------
# 3. GEMINI FUNCTION: WEEKLY STRATEGY DIGEST
# -------------------------------------------------------------
def generate_strategy_digest(feedback_rows):
    """
    feedback_rows = list of dicts from analyze_feedback()
    """
    prompt = f"""
You are a senior product strategist.

Given this week’s structured feedback data (JSON objects), produce:

1. Top recurring themes  
2. Emerging issues  
3. Sentiment trends  
4. 3–5 highest-impact action recommendations  
5. A short executive summary (5–7 sentences)

Write the output in clean Markdown.

Here is the data:
{json.dumps(feedback_rows, indent=2)}
"""

    response = model.generate_content(prompt)
    return response.text

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="Feedback-to-Strategy AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Feedback-to-Strategy AI Agent (Gemini-powered)")
st.write("Transform raw feedback into structured insights + weekly strategy digests.")

# Sidebar
mode = st.sidebar.radio("Choose Mode", ["Single Feedback", "Batch (CSV)", "Weekly Strategy Digest"])

# -------------------------------------------------------------
# SINGLE FEEDBACK MODE
# -------------------------------------------------------------
if mode == "Single Feedback":
    st.header("🔍 Analyze a Single Feedback Item")

    text = st.text_area("Enter feedback text")

    if st.button("Analyze"):
        if len(text.strip()) == 0:
            st.warning("Please enter feedback.")
        else:
            with st.spinner("Analyzing with Gemini…"):
                result = analyze_feedback(text)

            st.subheader("Results")
            st.json(result)

            st.metric("Confidence Score", result["confidence"])

# -------------------------------------------------------------
# BATCH MODE (CSV)
# -------------------------------------------------------------
elif mode == "Batch (CSV)":
    st.header("📂 Upload CSV of Feedback")

    file = st.file_uploader("Upload a CSV with a 'feedback' column")

    if file:
        df = pd.read_csv(file)

        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            if st.button("Process All"):
                all_results = []

                with st.spinner("Processing with Gemini…"):
                    for fb in df["feedback"].tolist():
                        result = analyze_feedback(fb)
                        all_results.append(result)

                output_df = pd.DataFrame(all_results)
                st.success("Completed!")
                st.dataframe(output_df)

                # Allow download
                csv = output_df.to_csv(index=False)
                st.download_button("Download Results", csv, "results.csv")

# -------------------------------------------------------------
# WEEKLY STRATEGY DIGEST
# -------------------------------------------------------------
elif mode == "Weekly Strategy Digest":
    st.header("📅 Generate Weekly Strategy Digest")

    file = st.file_uploader("Upload CSV of processed feedback", type=["csv"])

    if file:
        df = pd.read_csv(file)

        required = ["category", "sentiment", "summary", "key_phrases", "explanation", "confidence"]
        if not all(col in df.columns for col in required):
            st.error("This CSV must be the output of the Batch Mode analysis.")
        else:
            if st.button("Generate Digest"):
                rows = df.to_dict(orient="records")

                with st.spinner("Synthesizing weekly digest using Gemini…"):
                    digest = generate_strategy_digest(rows)

                st.markdown("## 📘 Weekly Strategy Digest")
                st.markdown(digest)

                st.download_button(
                    "Download Markdown", digest, file_name="weekly_digest.md"
                )
