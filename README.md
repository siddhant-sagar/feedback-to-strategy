# Feedback-to-Strategy AI Agent

A Human–AI system that transforms raw user feedback into structured, actionable product insights.

---

## Overview

Feedback-to-Strategy is a **Streamlit-based application** with two possible workflows:

1. **Direct AI Mode (app.py)**  
   - Processes feedback directly using Gemini/OpenAI APIs from the Streamlit app.  
   - Ideal for small teams or quick analysis.

2. **Agentic Mode (Streamlit + n8n backend)**  
   - Uses an **n8n workflow** as an AI agent backend.  
   - Handles single or batch feedback, computes confidence scores, stores structured data, and generates weekly strategic digests automatically.  
   - Supports automation, cron-based weekly digests, and dashboard visualizations.

It converts messy, unstructured feedback (surveys, reviews, support tickets, CSVs) into:

- **Categories:** Bug / Feature / UX / Performance / Other  
- **Summaries of key themes**  
- **Sentiment scores**  
- **Confidence scores + explanations**  
- **Trend analysis and dashboards**  
- **Weekly strategic digests for product teams**

The goal: turn feedback chaos into clarity, with transparent, explainable insights aligned with Human–AI Interaction (HAI) principles.

---

## Key Features

### Real-time Feedback Analysis (Direct AI Mode)

- Paste text or upload CSV files  
- Auto-classification (bug/feature/UX/performance/docs)  
- Theme summarization  
- Sentiment analysis  
- Confidence scoring  
- AI-generated reasoning  

### Weekly Strategy Digest (Agentic Mode)

- Automatically synthesizes the past 7 days of feedback into:  
  - Recurring patterns  
  - Emerging issues  
  - Sentiment shifts  
  - Prioritized recommendations  
- Delivery-ready Markdown digest  
- Fully automated via n8n workflow + AI agent  

### Human–AI Interaction Principles

- **Transparency:** confidence scores + keyword highlights  
- **Controllability:** users can override tags  
- **Error tolerance:** model surfaces uncertainty  
- **Understandability:** plain-English explanations  
- **Feedback loops:** corrections improve future insights  

---

## Architecture

### Direct AI Mode (Quick, Local)

Streamlit UI
│
▼
Gemini/OpenAI API
│
▼
JSON Output → Streamlit Dashboard


### Agentic Mode (Automated, n8n Backend)

Streamlit UI
│
▼
Webhook → n8n Workflow
│
├─ AI Analysis (Gemini/OpenAI)
├─ Compute Confidence
├─ Store in Google Sheets / DB
└─ Weekly Digest (Cron) → Email / Slack
│
▼
Streamlit Dashboard (Interactive)


---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/feedback-to-strategy.git
cd feedback-to-strategy


### 2. Install dependencies
pip install -r frontend/requirements.txt
