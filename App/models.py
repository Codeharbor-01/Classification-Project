from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    .models-hero {
        padding: 2.8rem 2.6rem;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 24px;
        background: linear-gradient(125deg, #102c38 0%, #145b60 55%, #1d8175 100%);
        color: #f5fbf9;
    }
    .models-hero .eyebrow {
        color: #a9e7d6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .models-hero h1 {
        margin: 0.7rem 0 0.8rem;
        color: #f5fbf9;
        font-size: clamp(2.3rem, 5vw, 3.7rem);
        line-height: 1.08;
    }
    .models-hero p {
        max-width: 710px;
        margin: 0;
        color: #d0e5e1;
        font-size: 1.05rem;
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
    .model-card {
        height: 100%;
        min-height: 230px;
        padding: 1.35rem;
        border: 1px solid rgba(100, 125, 130, 0.2);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.035);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .model-card:hover,
    .model-card:focus-within {
        transform: translateY(-4px);
        border-color: rgba(37, 132, 118, 0.55);
        box-shadow: 0 12px 28px rgba(16, 44, 56, 0.12);
    }
    .model-tag {
        color: #258476;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .model-card h3 {
        margin: 0.5rem 0;
        font-size: 1.12rem;
    }
    .model-card p {
        margin: 0 0 0.9rem;
        color: #718087;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .score-guide {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        text-align: left;
    }
    .score-guide th,
    .score-guide td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(100, 125, 130, 0.2);
        vertical-align: top;
    }
    .score-guide th {
        color: #258476;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .score-guide td:first-child {
        width: 150px;
        font-weight: 700;
        white-space: nowrap;
    }
    .score-context {
        margin: 1rem 0 0;
        color: #718087;
        line-height: 1.6;
    }
    .score-context + .score-context {
        margin-top: 0.65rem;
        padding-top: 0.65rem;
        border-top: 1px solid rgba(100, 125, 130, 0.2);
    }
    .score-context strong {
        color: #258476;
    }
    @media (prefers-reduced-motion: reduce) {
        .model-card {
            transition: none;
        }
        .model-card:hover,
        .model-card:focus-within {
            transform: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model_specs = [
    {
        "name": "Random Forest",
        "file": "RandomForestClassifier_scores.pkl",
        "tag": "01 · Ensemble",
        "description": (
            "Combines decision trees to capture non-linear relationships between "
            "clinical measurements and lifestyle indicators."
        ),
        "tuning": (
            "Grid searched 100–300 trees, maximum depth, minimum split size, "
            "and minimum leaf size."
        ),
    },
    {
        "name": "Logistic Regression",
        "file": "LogisticRegression_scores.pkl",
        "tag": "02 · Linear baseline",
        "description": (
            "A fast, interpretable linear classifier that estimates class "
            "probabilities from weighted input features."
        ),
        "tuning": "Grid searched regularization strength C from 0.01 to 100.",
    },
    {
        "name": "K-Nearest Neighbors",
        "file": "KNNClassifier_scores.pkl",
        "tag": "03 · Neighbor-based",
        "description": (
            "Classifies a record using nearby training examples; scaling and "
            "categorical encoding help make feature distances comparable."
        ),
        "tuning": (
            "Grid searched 3–15 neighbors, three distance metrics, "
            "and uniform or distance-based voting."
        ),
    },
]
metric_labels = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
}

scores_directory = Path(__file__).resolve().parents[1] / "Scores"
score_rows = []

for model in model_specs:
    score_path = scores_directory / model["file"]
    if not score_path.is_file():
        st.error(f"Missing evaluation scores: {score_path}")
        st.stop()

    scores = joblib.load(score_path)
    if not isinstance(scores, dict) or any(
        metric not in scores for metric in metric_labels
    ):
        st.error(f"Invalid score data in {score_path}.")
        st.stop()

    row = {"Model": model["name"]}
    for metric in metric_labels:
        try:
            value = float(scores[metric])
        except (TypeError, ValueError):
            st.error(f"Invalid {metric} score in {score_path}.")
            st.stop()
        if not 0 <= value <= 1:
            st.error(f"Expected {metric} in {score_path} to be between 0 and 1.")
            st.stop()
        row[metric] = value
    score_rows.append(row)

scores_df = pd.DataFrame(score_rows)

st.markdown(
    """
    <section class="models-hero">
        <div class="eyebrow">Model bench · Diabetes risk classification</div>
        <h1>Compare the classifiers.</h1>
        <p>
            Explore how three approaches perform on the same held-out data.
            Review their strengths, compare evaluation scores, and see how each
            model balances precision and recall.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Performance snapshot</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">The current leaderboard</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Rank the saved test-set results by the metric that matters most to your comparison.</p>',
    unsafe_allow_html=True,
)

leaderboard_metric = st.selectbox(
    "Rank models by",
    options=list(metric_labels),
    format_func=lambda metric: metric_labels[metric],
)
leaderboard = scores_df.sort_values(
    leaderboard_metric, ascending=False, kind="stable"
).reset_index(drop=True)
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
leaderboard_display = leaderboard.rename(
    columns={metric: metric_labels[metric] for metric in metric_labels}
)
for metric in metric_labels.values():
    leaderboard_display[metric] = leaderboard_display[metric].map(
        lambda value: f"{value:.1%}"
    )

leaderboard_columns = st.columns(3)
for column, (_, winner) in zip(leaderboard_columns, leaderboard.iterrows()):
    with column:
        st.metric(
            f"#{int(winner['Rank'])} · {winner['Model']}",
            f"{winner[leaderboard_metric]:.1%}",
            label_visibility="visible",
        )
st.dataframe(leaderboard_display, width="stretch", hide_index=True)

st.markdown('<div class="section-kicker">Score trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Performance across metrics</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Follow each model’s score across accuracy, precision, recall, and F1.</p>',
    unsafe_allow_html=True,
)
trend_data = scores_df.set_index("Model")[list(metric_labels)].T.rename(
    index=metric_labels
)
st.line_chart(trend_data, y_label="Score", x_label="Evaluation metric", height=340)
st.caption(
    "This compares the available evaluation metrics, not performance over time. "
    "Only one saved evaluation per model is available; historical score runs are not stored."
)

st.markdown('<div class="section-kicker">Model guide</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What each classifier does</div>', unsafe_allow_html=True)
model_columns = st.columns(3)
for column, model in zip(model_columns, model_specs):
    model_score = scores_df.loc[scores_df["Model"] == model["name"]].iloc[0]
    with column:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-tag">{model["tag"]}</div>
                <h3>{model["name"]}</h3>
                <p>{model["description"]}</p>
                <p><strong>Search space:</strong> {model["tuning"]}</p>
                <div class="model-tag">Accuracy · {model_score["accuracy"]:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-kicker">Evaluation notes</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How to read these scores</div>', unsafe_allow_html=True)
st.markdown(
    """
    <table class="score-guide">
        <thead>
            <tr><th>Metric</th><th>What it tells you</th></tr>
        </thead>
        <tbody>
            <tr>
                <td>Accuracy</td>
                <td>How many predictions were correct out of all test examples.</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>When the model predicts a risk class, how often that prediction is correct.</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>How many examples belonging to a risk class the model successfully identifies.</td>
            </tr>
            <tr>
                <td>F1 score</td>
                <td>A combined measure of precision and recall, useful when comparing both kinds of errors.</td>
            </tr>
        </tbody>
    </table>
    <p class="score-context">
        <strong>Reading the scale:</strong> All scores run from 0% to 100%; higher
        is better. Precision, recall, and F1 are weighted across risk classes, so an
        overall score may not reflect performance for every individual class.
    </p>
    <p class="score-context">
        <strong>How these were measured:</strong> Scores use the shared 20% held-out
        test set. Model settings were selected with five-fold cross-validation on
        the training data. These results are for comparing models in this project,
        not for clinical decision-making.
    </p>
    """,
    unsafe_allow_html=True,
)
