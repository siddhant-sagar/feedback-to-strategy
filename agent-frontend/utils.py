import streamlit as st
import pandas as pd
import json
import plotly.express as px

def render_dashboard(df, title="Feedback Insights Dashboard"):
    st.subheader(f"📊 {title}")
    col1, col2, col3 = st.columns(3)

    # Category Distribution
    with col1:
        st.markdown("**Category Distribution**")
        cat_counts = df["category"].value_counts()
        fig_cat = px.pie(
            names=cat_counts.index,
            values=cat_counts.values,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Feedback Categories"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Sentiment Distribution
    with col2:
        st.markdown("**Sentiment Distribution**")
        sentiment_counts = df["sentiment"].value_counts()
        fig_sent = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={"positive": "green", "neutral": "orange", "negative": "red"},
            title="Sentiment Overview"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # Confidence Score
    with col3:
        st.markdown("**Confidence Score**")
        avg_conf = round(df["confidence"].mean(), 2) if "confidence" in df.columns else 0
        st.metric("Average Confidence", avg_conf)
        if "confidence" in df.columns:
            fig_conf = px.histogram(df, x="confidence", nbins=10, title="Confidence Distribution", color_discrete_sequence=["blue"])
            st.plotly_chart(fig_conf, use_container_width=True)

    # Emerging Themes
    st.subheader("🌟 Emerging Themes / Key Phrases")
    all_phrases = []
    if "key_phrases" in df.columns:
        for kp in df["key_phrases"]:
            try:
                phrases = json.loads(kp.replace("'", '"')) if isinstance(kp, str) else kp
                if isinstance(phrases, list):
                    all_phrases.extend(phrases)
            except:
                pass
    if all_phrases:
        phrase_counts = pd.Series(all_phrases).value_counts().head(15)
        fig_phrases = px.bar(
            x=phrase_counts.values,
            y=phrase_counts.index,
            orientation="h",
            title="Top Key Phrases",
            color=phrase_counts.values,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_phrases, use_container_width=True)
