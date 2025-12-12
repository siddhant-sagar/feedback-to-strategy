import streamlit as st
import pandas as pd
import json
import time
import random
import google.generativeai as genai

# -------------------------------------------------------------
# 1. CONFIGURE GEMINI
# -------------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "models/gemini-1.5-flash"   # safest model for streamlit cloud
model = genai.GenerativeModel(MODEL_NAME)


# -------------------------------------------------------------
# 2. FALLBACK CLASSIFIER (no API needed)
# -------------------------------------------------------------
def fallback_classifier(text: str) -> dict:
    """A simple keyword-based backup classifier if Gemini quota is exceeded."""

    t = text.lower()

    if any(k in t for k in ["bug", "error", "crash", "broken"]):
        return {
            "category": "bug",
            "sentiment": "negative",
            "summary": text[:120] + "...",
            "key_phrases": ["system issue", "bug detected"],
            "explanation": "Fallback keyword detection.",
            "confidence": 0.45,
        }

    if any(k in t for k in ["add", "feature", "should have"]):
        return {
            "category": "feature_request",
            "sentiment": "neutral",
            "summary": text[:120] + "...",
            "key_phrases": ["feature request"],
            "explanation": "Fallback keyword detection.",
            "confidence": 0.45,
        }

    if any(k in t for k in ["slow", "confusing", "hard", "unclear", "ugly"]):
        return {
            "category": "ux_issue",
            "sentiment": "negative",
            "summary": text[:120] + "...",
            "key_phrases": ["ux friction"],
            "explanation": "Fallback keyword detection.",
            "confidence": 0.45,
        }

    return {
        "category": "other",
        "sentiment": "neutral",
        "summary": text[:120] + "...",
        "key_phrases": [],
        "explanation": "Fallback default.",
        "confidence": 0.30,
    }


# -------------------------------------------------------------
# 3. SAFE GEMINI CALL → retries + throttling + graceful fallback
# -------------------------------------------------------------
def call_gemini(prompt: str) -> str:
    """Calls Gemini safely with retries, throttle, and rate limit handling."""

    for attempt in range(3):
        try:
            # throttle BEFORE request
            time.sleep(random.uniform(1.5, 3.5))

            response = model.generate_content(prompt)

            # throttle AFTER request
            time.sleep(random.uniform(1.2, 2.5))

            return response.text

        except Exception as e:
            msg = str(e).lower()

            # retry on rate limits
            if "429" in msg or "quota" in msg:
                wait = random.uniform(4, 7)
                st.warning(f"⚠️ Rate limit hit. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue

            # fail-fast on non-retryable errors
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
        return fallback_classifier(text)

    try:
        data = json.loads(raw)
    except:
        data = fallback_classifier(text)

    # ---------------------------
    # RULE-BASED CONFIDENCE SCORE
    # ---------------------------
    confidence = 0.0

    # key phrases (0.4 weight)
    kp = len(data.get("key_phrases", []))
    confidence += min(kp / 6, 1) * 0.4

    # sentiment clarity (0.4 weight)
    if data.get("sentiment") in ["positive", "negative"]:
        confidence += 0.4
    else:
        confidence += 0.2

    # category presence (0.2 weight)
    if data.get("category") not in ["other", None, ""]:
        confidence += 0.2

    data["confidence"] = round(confidence, 2)
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
# 6. STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="Feedback-to-Strategy AI", layout="wide")

st.title("📊 Feedback-to-Strategy AI Agent (Gemini-Powered)")
st.write("Turn raw user feedback into structured insights + weekly strategy reports.")

mode = st.sidebar.radio(
    "Mode",
    ["Single Feedback", "Batch (CSV)", "Weekly Strategy Digest"]
)


# -------------------------------------------------------------
# SINGLE FEEDBACK MODE
# -------------------------------------------------------------
if mode == "Single Feedback":
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


# -------------------------------------------------------------
# BATCH CSV MODE
# -------------------------------------------------------------
elif mode == "Batch (CSV)":
    st.header("📂 Batch Feedback Processing")
    file = st.file_uploader("Upload CSV (must contain a 'feedback' column)")

    if file:
        df = pd.read_csv(file)

        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            if st.button("Process All"):
                results = []

                with st.spinner("Running batch analysis…"):
                    for fb in df["feedback"].tolist():
                        results.append(analyze_feedback(fb))

                out_df = pd.DataFrame(results)

                st.success("Completed!")
                st.dataframe(out_df)

                st.download_button(
                    "Download Results CSV",
                    out_df.to_csv(index=False),
                    "results.csv"
                )


# -------------------------------------------------------------
# WEEKLY STRATEGY DIGEST MODE
# -------------------------------------------------------------
elif mode == "Weekly Strategy Digest":
    st.header("📅 Weekly Strategy Digest Generator")

    file = st.file_uploader("Upload processed CSV from Batch mode")

    required = ["category", "sentiment", "summary", "key_phrases", "explanation", "confidence"]

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

                st.download_button(
                    "Download Digest (Markdown)",
                    digest,
                    "weekly_digest.md"
                )

# -------------------------------------------------------------
# DASHBOARD MODE
# -------------------------------------------------------------
elif mode == "Dashboard":
    st.header("📊 Feedback Insights Dashboard")
    file = st.file_uploader("Upload processed CSV from Batch mode")

    required = ["category", "sentiment", "summary", "key_phrases", "explanation", "confidence"]

    if file:
        df = pd.read_csv(file)

        if not all(col in df.columns for col in required):
            st.error("CSV must be the output of Batch Mode.")
        else:
            st.success("CSV loaded successfully!")

            # ---------------------------
            # CATEGORY DISTRIBUTION
            # ---------------------------
            st.subheader("1️⃣ Category Distribution")
            cat_counts = df["category"].value_counts()
            st.bar_chart(cat_counts)

            # ---------------------------
            # SENTIMENT DISTRIBUTION
            # ---------------------------
            st.subheader("2️⃣ Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # ---------------------------
            # EMERGING THEMES
            # ---------------------------
            st.subheader("3️⃣ Emerging Themes")
            # Flatten key_phrases list
            all_phrases = []
            for kp in df["key_phrases"]:
                try:
                    phrases = json.loads(kp.replace("'", '"')) if isinstance(kp, str) else kp
                    if isinstance(phrases, list):
                        all_phrases.extend(phrases)
                except:
                    pass
            phrase_counts = pd.Series(all_phrases).value_counts().head(10)
            st.table(phrase_counts)

            # ---------------------------
            # CONFIDENCE SCORE
            # ---------------------------
            st.subheader("4️⃣ Confidence Scores")
            st.metric("Average Confidence", round(df["confidence"].mean(), 2))
            st.line_chart(df["confidence"])

            # Optional download of dashboard summary
            summary_text = f"""
            Category Counts:\n{cat_counts.to_dict()}\n
            Sentiment Counts:\n{sentiment_counts.to_dict()}\n
            Top Key Phrases:\n{phrase_counts.to_dict()}\n
            Average Confidence: {round(df["confidence"].mean(), 2)}
            """
            st.download_button(
                "Download Dashboard Summary",
                summary_text,
                "dashboard_summary.txt"
            )
