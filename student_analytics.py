import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


REQUIRED_COLUMNS = [
    "Student Name",
    "Date",
    "Subject",
    "Score",
    "Engagement Level",
]


# -----------------------------
# Page configuration & styling
# -----------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
)


def inject_custom_css() -> None:
    """Apply a minimalist / glassmorphism-inspired style."""
    st.markdown(
        """
        <style>
        /* Global */
        .main {
            background: radial-gradient(circle at top left, #fdfbff 0, #eef3ff 40%, #e5f4ff 100%);
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-right: 1px solid rgba(255, 255, 255, 0.4);
        }

        /* Glass cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.82);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.65);
            box-shadow:
                0 18px 45px rgba(15, 23, 42, 0.08),
                0 0 0 1px rgba(148, 163, 184, 0.08);
        }

        /* Section headings */
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f172a;
            letter-spacing: 0.02em;
            margin-bottom: 0.4rem;
        }

        .section-subtitle {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.8rem;
        }

        /* Metrics */
        .metric-label {
            font-size: 0.8rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #111827;
        }

        /* Subtle badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            background: rgba(59, 130, 246, 0.09);
            color: #1d4ed8;
            border: 1px solid rgba(59, 130, 246, 0.22);
            margin-left: 0.5rem;
        }

        /* Risk table tweaks */
        .risk-note {
            font-size: 0.8rem;
            color: #6b7280;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# -----------------------------
# Synthetic data generation
# -----------------------------

@st.cache_data(show_spinner=False)
def generate_synthetic_data(
    n_students: int = 30,
    months: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic performance & engagement data."""
    rng = np.random.default_rng(seed)

    # Generate student names
    first_names = [
        "Ava",
        "Ethan",
        "Mia",
        "Liam",
        "Sophia",
        "Noah",
        "Isabella",
        "Lucas",
        "Emma",
        "Oliver",
        "Amelia",
        "Elijah",
        "Harper",
        "James",
        "Charlotte",
        "Benjamin",
        "Evelyn",
        "Henry",
        "Abigail",
        "Alexander",
        "Emily",
        "Michael",
        "Elizabeth",
        "Daniel",
        "Sofia",
        "Matthew",
        "Avery",
        "Jack",
        "Scarlett",
        "Logan",
        "Grace",
    ]
    last_names = [
        "Kim",
        "Patel",
        "Garcia",
        "Nguyen",
        "Smith",
        "Johnson",
        "Hernandez",
        "Chen",
        "Brown",
        "Lopez",
        "Davis",
        "Martinez",
        "Wilson",
        "Anderson",
        "Thomas",
    ]

    students = []
    for i in range(n_students):
        students.append(
            f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        )

    # Date range over the last `months` months (around 26 weeks of weekly data)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(weeks=4 * months)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="W")

    subjects = ["Math", "English"]
    records = []

    for student in students:
        # Base ability & engagement
        base_score = rng.normal(75, 8)
        engagement_base = rng.integers(2, 5)

        # Individual trend (slope) for the 6-month period
        score_trend = rng.normal(0, 3)  # can be slightly up or down

        for subject in subjects:
            # Subject-specific bias
            subj_delta = 4 if subject == "Math" else -2

            for t, date in enumerate(all_dates):
                # T is index in the time series
                noise = rng.normal(0, 7)
                score = base_score + subj_delta + score_trend * (t / len(all_dates)) * 10 + noise
                score = np.clip(score, 0, 100)

                # Engagement fluctuates but loosely correlated with score
                engagement_noise = rng.normal(0, 0.7)
                engagement = np.clip(
                    round(engagement_base + engagement_noise + (score - 70) / 25), 1, 5
                )

                records.append(
                    {
                        "Student Name": student,
                        "Date": date.date(),
                        "Subject": subject,
                        "Score": round(float(score), 1),
                        "Engagement Level": int(engagement),
                    }
                )

    df = pd.DataFrame(records)
    return df


# -----------------------------
# Analytics helpers
# -----------------------------


def load_uploaded_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a user-provided CSV or Excel file and validate the schema.

    Expected columns (case-sensitive):
    - Student Name
    - Date
    - Subject
    - Score
    - Engagement Level
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error(
                "Unsupported file type. Please upload a CSV or Excel (.xlsx) file."
            )
            return None
    except Exception as e:
        st.error(
            "We couldn't read that file. Please check that it is a valid CSV or Excel file."
        )
        st.caption(f"Technical detail: {e}")
        return None

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(
            "The uploaded file is missing one or more required columns: "
            + ", ".join(missing)
        )
        st.caption(
            "Expected columns are: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". You can download a clean template from the sidebar."
        )
        return None

    # Normalise and coerce types
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Engagement Level"] = pd.to_numeric(
        df["Engagement Level"], errors="coerce"
    ).astype("Int64")

    # Drop rows with no date or score
    before = len(df)
    df = df.dropna(subset=["Date", "Score"])
    if len(df) < before:
        st.info(
            f"{before - len(df)} row(s) were ignored because they were missing a date or score."
        )

    # Clip values into sensible ranges
    df["Score"] = df["Score"].clip(lower=0, upper=100)
    df["Engagement Level"] = df["Engagement Level"].clip(lower=1, upper=5)

    return df


def filter_data(
    df: pd.DataFrame,
    subject: Optional[str],
    students: Optional[List[str]],
) -> pd.DataFrame:
    filtered = df.copy()
    if subject and subject != "All":
        filtered = filtered[filtered["Subject"] == subject]
    if students:
        filtered = filtered[filtered["Student Name"].isin(students)]
    return filtered


def compute_risk_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate risk based on simple score trend over time for each student.

    We fit a linear trend of Score over time (per student, per subject combined),
    then classify those with a negative slope as at-risk. This is deliberately
    simple and pedagogically transparent for a PoC.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Student Name",
                "Average Score",
                "Average Engagement",
                "Score Trend (Δ / month)",
                "Risk Level",
            ]
        )

    df_sorted = df.sort_values(["Student Name", "Date"]).reset_index(drop=True)
    df_sorted["t"] = (
        (pd.to_datetime(df_sorted["Date"]) - pd.to_datetime(df_sorted["Date"]).min())
        .dt.days.astype(float)
        / 30.0
    )  # approximate months

    rows = []
    for student, grp in df_sorted.groupby("Student Name"):
        if len(grp) < 3:
            continue
        x = grp["t"].to_numpy()
        y = grp["Score"].to_numpy()

        # Simple linear regression y = a * x + b
        try:
            a, _b = np.polyfit(x, y, 1)
        except Exception:
            a = 0.0

        avg_score = grp["Score"].mean()
        avg_engagement = grp["Engagement Level"].mean()

        # Basic rule-of-thumb risk tiers
        if a <= -2.0:
            risk = "High"
        elif a <= -0.8:
            risk = "Moderate"
        else:
            risk = "Low"

        rows.append(
            {
                "Student Name": student,
                "Average Score": round(float(avg_score), 1),
                "Average Engagement": round(float(avg_engagement), 1),
                "Score Trend (Δ / month)": round(float(a), 2),
                "Risk Level": risk,
            }
        )

    risk_df = pd.DataFrame(rows)
    if risk_df.empty:
        return risk_df

    # Sort to show highest risk first
    risk_order = {"High": 0, "Moderate": 1, "Low": 2}
    risk_df["Risk Rank"] = risk_df["Risk Level"].map(risk_order)
    risk_df = risk_df.sort_values(["Risk Rank", "Score Trend (Δ / month)"]).drop(
        columns=["Risk Rank"]
    )
    return risk_df


def summarize_for_ai(df: pd.DataFrame) -> str:
    """Summarize key patterns from the current filtered data for prompting."""
    if df.empty:
        return "The current view contains no data."

    overall_avg = df["Score"].mean()
    overall_med = df["Score"].median()
    low_scores_pct = (df["Score"] < 60).mean() * 100
    high_scores_pct = (df["Score"] >= 85).mean() * 100
    avg_engagement = df["Engagement Level"].mean()

    subject_summary = (
        df.groupby("Subject")["Score"]
        .agg(["mean", "median"])
        .round(1)
        .reset_index()
        .to_dict(orient="records")
    )

    return (
        "Summary of current student performance data:\n"
        f"- Overall average score: {overall_avg:.1f}\n"
        f"- Overall median score: {overall_med:.1f}\n"
        f"- Percentage of scores below 60: {low_scores_pct:.1f}%\n"
        f"- Percentage of scores at or above 85: {high_scores_pct:.1f}%\n"
        f"- Average engagement level (1–5): {avg_engagement:.2f}\n"
        f"- Per-subject score summary: {subject_summary}\n"
    )


def generate_pedagogical_suggestions_placeholder(df: pd.DataFrame) -> List[str]:
    """Local fallback suggestions when Gemini is not configured."""
    if df.empty:
        return [
            "There is no data in the current view. Consider widening filters to see more students or dates.",
            "Once data is visible, use the trends to identify students who may benefit from targeted check-ins.",
        ]

    overall_avg = df["Score"].mean()
    avg_engagement = df["Engagement Level"].mean()
    low_engagement_ratio = (df["Engagement Level"] <= 2).mean()
    struggling_ratio = (df["Score"] < 60).mean()
    subject_means = df.groupby("Subject")["Score"].mean()

    suggestions: List[str] = []

    # Overall performance
    if overall_avg < 70:
        suggestions.append(
            "Overall performance is below 70%. Consider revisiting recent units with short, focused review activities and low-stakes formative checks."
        )
    else:
        suggestions.append(
            "Overall performance is reasonably strong. You can begin to introduce more complex, application-style tasks while still spot-checking fundamentals."
        )

    # Engagement
    if low_engagement_ratio > 0.25:
        suggestions.append(
            "A notable share of students show low engagement (levels 1–2). Incorporate quick interactive elements (think-pair-share, live polls, exit tickets) to re-energize lessons."
        )

    if avg_engagement >= 3.5:
        suggestions.append(
            "Engagement levels are generally healthy. Preserve routines that are working (clear success criteria, predictable lesson flow) while experimenting with student choice in tasks."
        )

    # Struggling learners
    if struggling_ratio > 0.2:
        suggestions.append(
            "Several students are scoring below 60%. Create a short list of focus learners and schedule 1:1 or small-group conferences to diagnose misconceptions and co-create goals."
        )

    # Subject differences
    if len(subject_means) >= 2:
        gap = subject_means.max() - subject_means.min()
        if gap >= 8:
            weaker_subject = subject_means.idxmin()
            suggestions.append(
                f"There is a clear gap between subjects; {weaker_subject} is trending lower. Audit recent {weaker_subject} tasks for cognitive load and provide additional scaffolded practice."
            )

    # Generic wrap-up
    suggestions.append(
        "Translate 1–2 of these insights into a concrete action plan for the next two weeks (e.g., which students to meet, which skill to re-teach, and how you will monitor impact)."
    )

    return suggestions[:5]


def generate_pedagogical_suggestions_with_gemini(df: pd.DataFrame) -> List[str]:
    """Attempt to call Gemini, falling back to the placeholder on any failure."""
    if genai is None:
        return generate_pedagogical_suggestions_placeholder(df)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return generate_pedagogical_suggestions_placeholder(df)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")

        summary = summarize_for_ai(df)
        prompt = f"""
You are an instructional designer and teacher coach.
You are looking at anonymised student performance data for a small class.

{summary}

Based only on this summary, propose 3–5 **actionable, concrete** pedagogical strategies
for the next 2–4 weeks. Focus on:
- how the teacher can respond instructionally,
- how to support struggling learners,
- how to maintain or increase engagement,
- and how to monitor whether the changes are working.

Respond as a short, numbered list in clear, plain English.
Do not mention that the data is synthetic or anonymised.
"""

        response = model.generate_content(prompt)
        text = response.text or ""
        # Split into bullet-like suggestions
        lines = [ln.strip("- ").strip() for ln in text.split("\n") if ln.strip()]
        # Keep only reasonably long lines
        lines = [ln for ln in lines if len(ln) > 12]
        return lines[:5] or generate_pedagogical_suggestions_placeholder(df)
    except Exception:
        return generate_pedagogical_suggestions_placeholder(df)


# -----------------------------
# Layout
# -----------------------------

st.sidebar.title("📊 Student Analytics")
st.sidebar.caption(
    "Data-informed progress tracking prototype for a small cohort of students."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Data source")
st.sidebar.caption(
    "Upload your own class data, or leave this empty to explore Demo Mode with synthetic data."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload class data (CSV or Excel)",
    type=["csv", "xlsx"],
    help=(
        "Use a CSV or Excel file with columns: "
        "Student Name, Date, Subject, Score, Engagement Level."
    ),
)

template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
st.sidebar.download_button(
    label="Download template (CSV)",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="student_performance_template.csv",
    mime="text/csv",
    help="Download a blank template in the required format. You can open it in Excel and paste your class records.",
)

st.sidebar.markdown("---")

data_mode = "Demo mode · Synthetic data"

# Generate data, preferring user upload when valid
base_df = generate_synthetic_data()
if uploaded_file is not None:
    user_df = load_uploaded_data(uploaded_file)
    if user_df is not None and not user_df.empty:
        base_df = user_df
        data_mode = f"Uploaded data · {uploaded_file.name}"
    elif user_df is None:
        # An error message will already be shown; keep demo data as a safe fallback.
        data_mode = "Demo mode · Synthetic data (upload invalid)"

st.sidebar.subheader("Filters")
st.sidebar.caption("Use these to explore patterns for specific subjects or learners.")

all_subjects = ["All"] + sorted(base_df["Subject"].unique().tolist())
subject_filter = st.sidebar.selectbox(
    "Subject",
    options=all_subjects,
    index=0,
    help="Filter the dashboard to a specific subject or view all subjects together.",
)

all_students = sorted(base_df["Student Name"].unique().tolist())
student_filter = st.sidebar.multiselect(
    "Students",
    options=all_students,
    default=[],
    help="Optionally focus on one or more specific students.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data source: {data_mode}. If you remove the upload, the app returns to Demo Mode."
)

filtered_df = filter_data(base_df, subject_filter, student_filter)


badge_label = (
    "Demo mode · Synthetic data" if "Demo mode" in data_mode else "Uploaded class data"
)

st.markdown(
    f"""
    <div class="glass-card">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <div class="metric-label">EDTECH POC</div>
                <h2 style="margin:0.15rem 0 0.2rem; font-size:1.9rem; font-weight:650; letter-spacing:0.02em; color:#020617;">
                    Student Performance Analytics Dashboard
                </h2>
                <p style="margin:0; font-size:0.9rem; color:#6b7280;">
                    A data-informed progress tracking proof of concept for a small cohort of learners.
                </p>
            </div>
            <div style="text-align:right;">
                <span class="badge">{badge_label}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Top metrics
# -----------------------------

with st.container():
    col1, col2, col3, col4 = st.columns(4)

    if filtered_df.empty:
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">No data</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-subtitle">Adjust filters to view student performance.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        time_range = (
            filtered_df["Date"].min().strftime("%b %d, %Y")
            + " – "
            + filtered_df["Date"].max().strftime("%b %d, %Y")
        )

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-label">Average score</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-value">{filtered_df["Score"].mean():.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-subtitle">Across all visible assessments.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-label">Engagement (1–5)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-value">{filtered_df["Engagement Level"].mean():.2f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-subtitle">Higher values indicate stronger learner engagement.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            at_or_above_85 = (filtered_df["Score"] >= 85).mean() * 100
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-label">High performers</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-value">{at_or_above_85:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-subtitle">Share of scores at or above 85%.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-label">Time window</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-value" style="font-size:1.05rem;">{time_range}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-subtitle">Automatically generated over the last six months.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown("")  # small vertical space


# -----------------------------
# Charts row
# -----------------------------

col_left, col_right = st.columns((2, 1))

with col_left:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Average score trend over time</div>
            <div class="section-subtitle">
                Line chart of mean score for each week in the selected view.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.info("No data available for the current filters. Try selecting another subject or more students.")
    else:
        df_line = (
            filtered_df.groupby("Date")["Score"]
            .mean()
            .reset_index()
            .sort_values("Date")
        )

        line_chart = (
            alt.Chart(df_line)
            .mark_line(point=True)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Score:Q", title="Average score"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Score:Q", title="Average score", format=".1f"),
                ],
            )
            .properties(height=340)
        )

        st.altair_chart(line_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Score distribution</div>
            <div class="section-subtitle">
                Distribution of individual assessment scores in the current view.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.info("No scores to show. Adjust filters to see the score distribution.")
    else:
        hist = (
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("Score:Q", bin=alt.Bin(maxbins=20), title="Score"),
                y=alt.Y("count():Q", title="Number of observations"),
                tooltip=[
                    alt.Tooltip("count():Q", title="Count"),
                ],
            )
            .properties(height=340)
        )

        st.altair_chart(hist, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Risk assessment
# -----------------------------

st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">Risk assessment by student</div>
        <div class="section-subtitle">
            Highlights learners whose scores show declining trends over the selected time period.
        </div>
    """,
    unsafe_allow_html=True,
)

risk_df = compute_risk_assessment(filtered_df)

if risk_df.empty:
    st.info("Not enough data to compute risk assessment for the current view.")
else:
    def _color_risk(val: str) -> str:
        if val == "High":
            return "background-color: rgba(239, 68, 68, 0.12); color: #b91c1c;"
        if val == "Moderate":
            return "background-color: rgba(245, 158, 11, 0.12); color: #92400e;"
        if val == "Low":
            return "background-color: rgba(16, 185, 129, 0.10); color: #047857;"
        return ""

    styled = risk_df.style.applymap(_color_risk, subset=["Risk Level"])
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<p class="risk-note">Risk levels are based on simple linear trends in scores over time and are intended only as a discussion starter, not a high-stakes label.</p>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# AI pedagogical insight
# -----------------------------

st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">AI pedagogical insight</div>
        <div class="section-subtitle">
            Generate 3–5 actionable teaching strategies based on the current data trends.
        </div>
    """,
    unsafe_allow_html=True,
)

use_gemini = st.checkbox(
    "Use Gemini API (if configured)",
    value=False,
    help="When checked, the app will attempt to call Gemini using the GEMINI_API_KEY environment variable. "
    "If unavailable, it will gracefully fall back to a local placeholder report.",
)

if st.button(
    "Generate AI report",
    help="Click to generate a short, human-readable summary of pedagogical next steps informed by the filtered data.",
):
    with st.spinner("Generating pedagogical suggestions..."):
        if use_gemini:
            suggestions = generate_pedagogical_suggestions_with_gemini(filtered_df)
        else:
            suggestions = generate_pedagogical_suggestions_placeholder(filtered_df)

    if not suggestions:
        st.warning("No suggestions could be generated for the current view.")
    else:
        st.markdown("#### Suggested next steps")
        for i, s in enumerate(suggestions, start=1):
            st.markdown(f"**{i}.** {s}")

st.markdown("</div>", unsafe_allow_html=True)

