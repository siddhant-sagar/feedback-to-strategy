import streamlit as st
import pandas as pd
import json
import time
import random
import plotly.express as px
import google.generativeai as genai

# -------------------------------------------------------------
# 1. CONFIGURE GEMINI
# -------------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_NAME = "models/gemini-1.5-flash"  # safest model for streamlit cloud
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------------------
# 2. FALLBACK CLASSIFIER (no API needed)
# -------------------------------------------------------------
def fallback_classifier(text: str) -> dict:
    t = text.lower()
    if any(k in t for k in ["bug", "error", "crash", "broken"]):
        return {"category": "bug", "sentiment": "negative", "summary": text[:120]+"...", 
                "key_phrases":["system issue","bug detected"], "explanation":"Fallback keyword detection", "confidence":0.45}
    if any(k in t for k in ["add", "feature", "should have"]):
        return {"category": "feature_request", "sentiment": "neutral", "summary": text[:120]+"...", 
                "key_phrases":["feature request"], "explanation":"Fallback keyword detection", "confidence":0.45}
    if any(k in t for k in ["slow", "confusing", "hard", "unclear", "ugly"]):
        return {"category": "ux_issue", "sentiment": "negative", "summary": text[:120]+"...", 
                "key_phrases":["ux friction"], "explanation":"Fallback keyword detection", "confidence":0.45}
    return {"category":"other", "sentiment":"neutral","summary":text[:120]+"...", "key_phrases":[], 
            "explanation":"Fallback default", "confidence":0.30}

# -------------------------------------------------------------
# 3. SAFE GEMINI CALL
# -------------------------------------------------------------
def call_gemini(prompt: str) -> str:
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1.5,3.5))
            response = model.generate_content(prompt)
            time.sleep(random.uniform(1.2,2.5))
            return response.text
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg:
                wait=random.uniform(4,7)
                st.warning(f"⚠️ Rate limit hit. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            return f"ERROR: {e}"
    return "ERROR: Max retries exceeded."

# -------------------------------------------------------------
# 4. ANALYZE SINGLE FEEDBACK
# -------------------------------------------------------------
def analyze_feedback(text: str) -> dict:
    prompt = f"""
    You are an AI feedback insights analyst.
    Analyze the feedback and return STRICT JSON with:
    - category: [bug, feature_request, ux_issue, performance, documentation, other]
    - sentiment: positive | neutral | negative
    - summary: 1–2 sentences
    - key_phrases: 3–6 strong signals
    - explanation: short reasoning
    Feedback: "{text}"
    Return ONLY JSON.
    """
    raw = call_gemini(prompt)
    if raw.startswith("ERROR"):
        data = fallback_classifier(text)
    else:
        try:
            data = json.loads(raw)
        except:
            data = fallback_classifier(text)

    # Confidence score
    confidence = 0.0
    kp = len(data.get("key_phrases", []))
    confidence += min(kp/6,1)*0.4
    confidence += 0.4 if data.get("sentiment") in ["positive","negative"] else 0.2
    confidence += 0.2 if data.get("category") not in ["other",None,""] else 0
    data["confidence"] = round(confidence,2)
    return data

# -------------------------------------------------------------
# 5. WEEKLY STRATEGY DIGEST
# -------------------------------------------------------------
def generate_strategy_digest(rows: list) -> str:
    prompt = f"""
    You are a senior product strategist.
    Analyze this week's structured feedback and produce:
    1. Top recurring themes
    2. Emerging issues
    3. Sentiment trends
    4. 3–5 high-impact recommendations
    5. A 5–7 sentence executive summary
    Return the output in clean MARKDOWN.
    Data:
    {json.dumps(rows, indent=2)}
    """
    raw = call_gemini(prompt)
    if raw.startswith("ERROR"):
        return """
        ## Weekly Digest (Fallback Mode)
        Gemini quota was exceeded — using backup summary.
        ### Themes
        - UX friction  
        - Bug reports  
        - Feature requests  
        ### Recommendations
        - Address UX consistency  
        - Fix top 2 bugs  
        - Prioritize one key feature request  
        A full digest will appear once API quota resets.
        """
    return raw

# -------------------------------------------------------------
# 6. DASHBOARD FUNCTION (Reusable)
# -------------------------------------------------------------
def render_dashboard(df, title="Feedback Insights Dashboard"):
    st.subheader(f"📊 {title}")

    col1, col2, col3 = st.columns(3)

    # Category Distribution
    with col1:
        st.markdown("**Category Distribution**")
        cat_counts = df["category"].value_counts()
        fig_cat = px.pie(names=cat_counts.index, values=cat_counts.values, color_discrete_sequence=px.colors.qualitative.Set2, title="Feedback Categories")
        st.plotly_chart(fig_cat, use_container_width=True)

    # Sentiment Distribution
    with col2:
        st.markdown("**Sentiment Distribution**")
        sentiment_counts = df["sentiment"].value_counts()
        fig_sent = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                          color=sentiment_counts.index,
                          color_discrete_map={"positive":"green","neutral":"orange","negative":"red"},
                          title="Sentiment Overview")
        st.plotly_chart(fig_sent, use_container_width=True)

    # Confidence Score
    with col3:
        st.markdown("**Confidence Score**")
        avg_conf = round(df["confidence"].mean(),2)
        st.metric("Average Confidence", avg_conf)
        fig_conf = px.histogram(df, x="confidence", nbins=10, title="Confidence Distribution", color_discrete_sequence=["blue"])
        st.plotly_chart(fig_conf, use_container_width=True)

    # Emerging Themes
    st.subheader("🌟 Emerging Themes / Key Phrases")
    all_phrases = []
    for kp in df["key_phrases"]:
        try:
            phrases = json.loads(kp.replace("'", '"')) if isinstance(kp,str) else kp
            if isinstance(phrases,list):
                all_phrases.extend(phrases)
        except:
            pass
    phrase_counts = pd.Series(all_phrases).value_counts().head(15)
    fig_phrases = px.bar(x=phrase_counts.values, y=phrase_counts.index, orientation="h",
                         title="Top Key Phrases", color=phrase_counts.values, color_continuous_scale="Viridis")
    st.plotly_chart(fig_phrases, use_container_width=True)

    # Optional download
    summary_text = f"""
    Category Counts:\n{cat_counts.to_dict()}\n
    Sentiment Counts:\n{sentiment_counts.to_dict()}\n
    Top Key Phrases:\n{phrase_counts.to_dict()}\n
    Average Confidence: {avg_conf}
    """
    st.download_button("Download Dashboard Summary", summary_text, "dashboard_summary.txt")

# -------------------------------------------------------------
# 7. STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="Feedback-to-Strategy AI", layout="wide")
st.title("📊 Feedback-to-Strategy AI Agent (Gemini-Powered)")
st.write("Turn raw user feedback into structured insights + weekly strategy reports.")

mode = st.sidebar.radio(
    "Mode",
    ["Single Feedback", "Batch (CSV)", "Weekly Strategy Digest"],
    index=1  # default to Batch mode
)

# -----------------------------
# Single Feedback Mode
# -----------------------------
if mode=="Single Feedback":
    st.header("🔍 Analyze Single Feedback")
    user_text = st.text_area("Enter feedback text")
    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter feedback.")
        else:
            with st.spinner("Processing with Gemini…"):
                result = analyze_feedback(user_text)
            st.subheader("Result")
            st.json(result)
            st.metric("Confidence Score", result["confidence"])

# -----------------------------
# Batch Mode
# -----------------------------
elif mode=="Batch (CSV)":
    st.header("📂 Batch Feedback Processing")
    file = st.file_uploader("Upload CSV (must contain a 'feedback' column)")
    if file:
        df = pd.read_csv(file)
        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            if st.button("Process All"):
                results=[]
                with st.spinner("Running batch analysis…"):
                    for fb in df["feedback"].tolist():
                        results.append(analyze_feedback(fb))
                out_df = pd.DataFrame(results)
                st.success("Completed!")
                st.dataframe(out_df)
                st.download_button("Download Results CSV", out_df.to_csv(index=False),"results.csv")

                # Render Dashboard
                render_dashboard(out_df, title="Batch Feedback Insights")

# -----------------------------
# Weekly Digest Mode
# -----------------------------
elif mode=="Weekly Strategy Digest":
    st.header("📅 Weekly Strategy Digest Generator")
    file = st.file_uploader("Upload processed CSV from Batch mode")
    required=["category","sentiment","summary","key_phrases","explanation","confidence"]
    if file:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in required):
            st.error("CSV must be the output of Batch Mode.")
        else:
            if st.button("Generate Digest"):
                rows = df.to_dict(orient="records")
                with st.spinner("Synthesizing weekly digest…"):
                    digest = generate_strategy_digest(rows)
                st.markdown("## 📘 Weekly Strategy Digest")
                st.markdown(digest)
                st.download_button("Download Digest (Markdown)", digest,"weekly_digest.md")

                # Render Dashboard
                render_dashboard(df, title="Weekly Feedback Insights")
