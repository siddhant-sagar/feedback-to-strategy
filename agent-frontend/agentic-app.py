import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
from utils import render_dashboard

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Feedback-to-Insights", layout="wide")
st.title("📊 Feedback-to-Insights Dashboard")
st.write("Submit feedback and visualize insights processed by n8n backend.")

N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL")

mode = st.sidebar.radio(
    "Mode",
    ["Single Feedback", "Batch CSV Upload", "Dashboard"],
    index=0
)

# -----------------------------
# Single Feedback Submission
# -----------------------------
if mode == "Single Feedback":
    st.header("🔍 Submit Single Feedback")
    feedback_text = st.text_area("Enter feedback text")

    if st.button("Submit Feedback"):
        if not feedback_text.strip():
            st.warning("Please enter feedback.")
        else:
            with st.spinner("Sending feedback to backend..."):
                response = requests.post(
                    N8N_WEBHOOK_URL,
                    json={"feedback_text": feedback_text}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Feedback processed successfully!")
                    st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

# -----------------------------
# Batch CSV Upload
# -----------------------------
elif mode == "Batch CSV Upload":
    st.header("📂 Upload CSV of Feedback")
    uploaded_file = st.file_uploader("CSV file must contain 'feedback_text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "feedback_text" not in df.columns:
            st.error("CSV missing required column: 'feedback_text'")
        else:
            if st.button("Submit All Feedback"):
                results = []
                for fb in df["feedback_text"]:
                    response = requests.post(
                        N8N_WEBHOOK_URL,
                        json={"feedback_text": fb}
                    )
                    if response.status_code == 200:
                        results.append(response.json())
                if results:
                    out_df = pd.DataFrame(results)
                    st.success("Batch processed successfully!")
                    st.dataframe(out_df)
                    st.download_button("Download Results CSV", out_df.to_csv(index=False), "batch_results.csv")

                    # Render Dashboard
                    render_dashboard(out_df, title="Batch Feedback Insights")

# -----------------------------
# Dashboard Mode
# -----------------------------
elif mode == "Dashboard":
    st.header("📊 Dashboard Insights")
    st.info("Pull structured feedback data from Google Sheets / DB to visualize here.")
    
    # Example: Pull from Google Sheets CSV export link
    sheet_url = st.text_input("Enter Google Sheets CSV export URL")
    if sheet_url and st.button("Load Data"):
        df = pd.read_csv(sheet_url)
        if "category" in df.columns and "sentiment" in df.columns:
            render_dashboard(df, title="Live Feedback Insights")
        else:
            st.error("CSV must contain at least 'category' and 'sentiment' columns.")
