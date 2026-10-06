from pathlib import Path

import pandas as pd
import streamlit as st

df = pd.read_csv(r'Dataset/Cleaned_dataset.csv')

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 3.1rem 3rem;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 24px;
        background: linear-gradient(125deg, #102c38 0%, #145b60 55%, #1d8175 100%);
        color: #f5fbf9;
    }
    .hero::after {
        content: "＋";
        position: absolute;
        right: 4%;
        top: -72px;
        color: rgba(255, 255, 255, 0.08);
        font-size: 300px;
        font-weight: 200;
        line-height: 1;
    }
    .eyebrow {
        color: #a9e7d6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .hero h1 {
        max-width: 720px;
        margin: 0.7rem 0 0.8rem;
        color: #f5fbf9;
        font-size: clamp(2.4rem, 5vw, 4rem);
        line-height: 1.08;
    }
    .hero p {
        max-width: 670px;
        margin: 0;
        color: #d0e5e1;
        font-size: 1.08rem;
        line-height: 1.7;
    }
    .section-kicker {
        margin: 2.7rem 0 0.35rem;
        color: #258476;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 0.45rem;
        font-size: 1.85rem;
        font-weight: 700;
    }
    .section-copy {
        margin: 0 0 1.25rem;
        color: #718087;
        line-height: 1.65;
    }
    .info-card {
        height: 100%;
        min-height: 155px;
        padding: 1.2rem 1.25rem;
        border: 1px solid rgba(100, 125, 130, 0.2);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.035);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .info-card:hover,
    .info-card:focus-within,
    .model-card:hover,
    .model-card:focus-within {
        transform: translateY(-4px);
        border-color: rgba(37, 132, 118, 0.55);
        box-shadow: 0 12px 28px rgba(16, 44, 56, 0.12);
    }
    .info-card .icon {
        font-size: 1.4rem;
    }
    .info-card h3 {
        margin: 0.65rem 0 0.4rem;
        font-size: 1.05rem;
    }
    .info-card p {
        margin: 0;
        color: #718087;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .st-key-dashboard-metrics [data-testid="column"] {
        position: relative;
    }
    .st-key-dashboard-metrics [data-testid="column"]:not(:first-child)::before {
        position: absolute;
        top: 22%;
        left: 0;
        height: 56%;
        border-left: 1px solid rgba(100, 125, 130, 0.28);
        content: "";
    }
    .model-card {
        height: 100%;
        padding: 1.35rem;
        border: 1px solid rgba(100, 125, 130, 0.2);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.035);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .model-card h3 {
        margin: 0.45rem 0;
        font-size: 1.08rem;
    }
    .model-card p {
        margin: 0;
        color: #718087;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .model-tag {
        color: #258476;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .note {
        padding: 1rem 1.1rem;
        border-left: 4px solid #e3ad55;
        border-radius: 0 12px 12px 0;
        background: rgba(227, 173, 85, 0.1);
        color: #718087;
        line-height: 1.6;
    }
    @media (prefers-reduced-motion: reduce) {
        .info-card,
        .model-card {
            transition: none;
        }
        .info-card:hover,
        .info-card:focus-within,
        .model-card:hover,
        .model-card:focus-within {
            transform: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Machine learning · Preventive health</div>
        <h1>Understand diabetes risk, one insight at a time.</h1>
        <p>
            An interactive classification project exploring how health measurements,
            medical history, and lifestyle factors can help estimate diabetes risk.
            Explore the data, compare models, and try a sample prediction.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">At a glance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">The project in numbers</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">A snapshot of the cleaned dataset used to train and explore the classifiers.</p>',
    unsafe_allow_html=True,
)

metric_container = st.container(border=True, key="dashboard-metrics")
metric_columns = metric_container.columns(4)
with metric_columns[0]:
    st.metric("Patient records", f"{len(df):,}")
with metric_columns[1]:
    st.metric("Dataset fields", f"{len(df.columns):,}")
with metric_columns[2]:
    st.metric("Risk classes", f"{df['Diabetes_Risk'].nunique():,}")
with metric_columns[3]:
    st.metric("Classifiers", "3")

st.markdown('<div class="section-kicker">Why this project</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Objectives & goals</div>', unsafe_allow_html=True)
objective_columns = st.columns(3)
objectives = [
    (
        "🎯",
        "Estimate risk",
        "Train classification models to categorize records into diabetes risk groups using patient and lifestyle indicators.",
    ),
    (
        "🧪",
        "Compare approaches",
        "Evaluate Logistic Regression, K-Nearest Neighbors, and Random Forest with consistent train/test data and metrics.",
    ),
    (
        "💡",
        "Make results explorable",
        "Present model performance, data patterns, and sample predictions through an approachable interactive application.",
    ),
]
for column, (icon, title, description) in zip(objective_columns, objectives):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="icon">{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-kicker">Model bench</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Three ways to classify risk</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Each model is paired with a preprocessing pipeline and tuned with five-fold cross-validation.</p>',
    unsafe_allow_html=True,
)
model_columns = st.columns(3)
models = [
    (
        "01 · Linear baseline",
        "Logistic Regression",
        "A clear, efficient baseline that estimates class probabilities from a weighted combination of input features.",
    ),
    (
        "02 · Neighbor-based",
        "K-Nearest Neighbors",
        "Classifies a record by comparing it with nearby examples after scaling numeric features and encoding categories.",
    ),
    (
        "03 · Ensemble",
        "Random Forest",
        "Combines decision trees to capture non-linear relationships across clinical and lifestyle indicators.",
    ),
]
for column, (tag, title, description) in zip(model_columns, models):
    with column:
        st.markdown(
            f'<div class="model-card"><div class="model-tag">{tag}</div><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-kicker">Data & methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What goes into a prediction?</div>', unsafe_allow_html=True)
data_columns = st.columns([1.1, 0.9])
with data_columns[0]:
    st.markdown(
        """
        <div class="info-card">
            <h3>Signals represented in the dataset</h3>
            <p>
                The cleaned records include demographic details, body measurements,
                glucose and lipid markers, blood pressure, family and medical history,
                activity, diet, sleep, and other lifestyle indicators.
            </p>
            <br>
            <p>
                Numeric features are standardized; ordered categories are ordinally
                encoded, and nominal categories use binary encoding. The target is
                <strong>Diabetes_Risk</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with data_columns[1]:
    risk_counts = (
        df["Diabetes_Risk"]
        .value_counts()
        .rename_axis("Risk category")
        .rename("Records")
    )
    st.markdown("#### Risk category distribution")
    st.bar_chart(risk_counts, color="#258476", height=230)

st.markdown("#### Sample records")

st.dataframe(df.sample(5), width="stretch", hide_index=True)
st.caption("Five example rows from the cleaned dataset. Values are provided for project exploration, not clinical guidance.")

st.markdown('<div class="section-kicker">Built with</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Libraries & workflow</div>', unsafe_allow_html=True)
library_columns = st.columns(5)
libraries = [
    ("🐍", "Python", "Application and model development"),
    ("📊", "Pandas", "Tabular data loading and inspection"),
    ("🧠", "scikit-learn", "Preprocessing, model selection, and evaluation"),
    ("🏷️", "category_encoders", "Converting categorical variables into numeric formats"),
    ("🧰", "joblib", "Saving and loading trained models")
]
for column, (icon, title, description) in zip(library_columns, libraries):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="icon">{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            
            <style>
                .info-card{{
                    height:220px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <br>
    <div class="note">
        <strong>Important:</strong> This project is for educational and exploratory use only.
        Predictions are not a diagnosis and should not replace advice from a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)