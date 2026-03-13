# student-performance-analytics
This is a **Student Performance Analytics Dashboard** built with **Streamlit** as a proof of concept for **data-informed progress tracking** in an EdTech / instructional design context.

It automatically generates a synthetic dataset of student performance and provides:

- Average score trends over time
- Score distribution visualization
- A risk assessment table that highlights students with declining performance
- An **AI Pedagogical Insight** report generator (placeholder or Gemini API)

## Features

- **Synthetic data generation**
  - 30 mock students
  - 6 months of performance data
  - Subjects: Math and English
  - Fields: Student Name, Date, Subject, Score (0–100), Engagement Level (1–5)

- **Visualizations**
  - Line chart for average score trend over time
  - Distribution plot of student scores
  - Risk assessment table for students with declining performance

- **AI Pedagogical Insight**
  - A `Generate AI Report` button produces 3–5 actionable teaching suggestions
  - If a Gemini API key is configured, it will use Gemini for generation
  - Otherwise, it falls back to a local placeholder report

- **UI / UX**
  - Clean, minimalist / glassmorphism-inspired design
  - Sidebar filters for Subject and Student
  - English UI and tooltips

## Installation

1. Create and activate a virtual environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional: Configure Gemini API

To enable real AI generation via Gemini:

1. Create a Gemini API key from Google AI Studio.
2. Set the environment variable before running the app (PowerShell example):

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

If `GEMINI_API_KEY` is not set or Gemini is unavailable, the app will gracefully fall back to a built-in placeholder set of suggestions.

## Running the App

From the project root:

```bash
streamlit run student_analytics.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`) in your browser.

## Notes

- This is designed as a **portfolio-ready PoC** to demonstrate **data-informed progress tracking** and analytics thinking for student performance.
- All data is synthetic and safe to screen-share or include in a resume / portfolio.
