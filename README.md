#  **Feedback-to-Strategy AI Agent
**

## A Human–AI system that transforms raw user feedback into structured, actionable product insights.

**Overview
**
Feedback-to-Strategy is a Streamlit-based application with two possible workflows:

1. Direct AI Mode (app.py)
- Processes feedback directly using Gemini/OpenAI APIs from the Streamlit app.
- Ideal for small teams or quick analysis.

2. Agentic Mode (Streamlit + n8n backend)
- Uses an n8n workflow as an AI agent backend.
- Handles single or batch feedback, computes confidence scores, stores structured data, and generates weekly strategic digests automatically.
- Supports automation, cron-based weekly digests, and dashboard visualizations.

It converts messy, unstructured feedback (surveys, reviews, support tickets, CSVs) into:
- Categories: Bug / Feature / UX / Performance / Other
- Summaries of key themes
- Sentiment scores
- Confidence scores + explanations
- Trend analysis and dashboards
- Weekly strategic digests for product teams

**The goal:** turn feedback chaos into clarity, with transparent, explainable insights aligned with Human–AI Interaction (HAI) principles.
