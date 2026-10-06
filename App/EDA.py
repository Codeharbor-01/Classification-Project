from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.title("Exploratory Data Analysis")
st.markdown(
    """
    Explore how the dataset changes during cleaning, inspect patterns in the
    patient records, and review the main observations before using the data
    for model training. The analysis below uses the cleaned dataset unless
    otherwise stated.
    """
)

dataset_directory = Path(__file__).resolve().parents[1] / "Dataset"
raw_path = dataset_directory / "diabetes_risk_prediction_dataset.csv"
cleaned_path = dataset_directory / "Cleaned_dataset.csv"

if not raw_path.is_file() or not cleaned_path.is_file():
    st.error("The raw or cleaned dataset file could not be found in the Dataset folder.")
    st.stop()

raw_data = pd.read_csv(raw_path)
cleaned_data = pd.read_csv(cleaned_path)

# The cleaned CSV contains a saved DataFrame index; it is not a dataset feature.
cleaned_data = cleaned_data.drop(columns=["Unnamed: 0"], errors="ignore")

required_columns = {
    "Diabetes_Risk",
    "Diabetes_Risk_Score",
    "Physical_Activity_Level",
    "Diet_Quality",
    "Family_History_Diabetes",
    "Age",
    "BMI",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
}
if not required_columns.issubset(cleaned_data.columns):
    st.error("The cleaned dataset is missing columns required for this analysis.")
    st.stop()

tabs = st.tabs(["Overview", "Analysis", "Findings"])

with tabs[0]:
    st.subheader("Dataset before and after cleaning")
    st.write(
        "The table compares the original file with the cleaned file. File sizes "
        "are shown in decimal megabytes. The cleaned file's saved row index is "
        "excluded from the column count."
    )

    raw_summary = {
        "Dataset": "Original (uncleaned)",
        "Rows": len(raw_data),
        "Columns": len(raw_data.columns),
        "Missing cells": int(raw_data.isna().sum().sum()),
        "Rows with missing values": int(raw_data.isna().any(axis=1).sum()),
        "Duplicate rows": int(raw_data.duplicated().sum()),
        "File size": f"{raw_path.stat().st_size / 1_000_000:.2f} MB",
    }
    cleaned_summary = {
        "Dataset": "Cleaned",
        "Rows": len(cleaned_data),
        "Columns": len(cleaned_data.columns),
        "Missing cells": int(cleaned_data.isna().sum().sum()),
        "Rows with missing values": int(cleaned_data.isna().any(axis=1).sum()),
        "Duplicate rows": int(cleaned_data.duplicated().sum()),
        "File size": f"{cleaned_path.stat().st_size / 1_000_000:.2f} MB",
    }
    st.dataframe(
        pd.DataFrame([raw_summary, cleaned_summary]).set_index("Dataset"),
        width="stretch",
    )
    st.info(
        f"The cleaned dataset contains {len(cleaned_data):,} records with "
        f"{int(cleaned_data.isna().sum().sum()):,} missing cells. "
        "The saved row-index column is omitted from the analysis."
    )

with tabs[1]:
    st.subheader("Patterns in the cleaned data")
    st.write(
        "Charts are shown one at a time at full width for easier reading. "
        "Together they show class balance, feature distributions, relationships "
        "between measurements, group comparisons, and numeric correlations."
    )

    sns.set_theme(style="whitegrid")
    risk_order = ["Low", "Moderate", "High"]
    risk_palette = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#e76f51"}

    st.markdown("#### Risk-label distribution")
    risk_counts = cleaned_data["Diabetes_Risk"].value_counts().reindex(risk_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=cleaned_data, x="Diabetes_Risk", order=risk_order,
        hue="Diabetes_Risk", palette=risk_palette, legend=False, ax=ax,
    )
    ax.set(title="Number of records in each risk group", xlabel="Risk group", ylabel="Records")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("#### Share of each risk group")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        risk_counts,
        labels=[f"{risk} ({count / len(cleaned_data):.1%})" for risk, count in risk_counts.items()],
        colors=[risk_palette[risk] for risk in risk_order],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title("Risk groups as a share of the cleaned dataset")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.divider()

    st.markdown("#### Histogram: HbA1c by risk group")
    st.write(
        "This histogram compares the distribution of HbA1c measurements in each "
        "risk group. The curves help show where values are concentrated and how "
        "much the groups overlap."
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(
        data=cleaned_data,
        x="HbA1c",
        hue="Diabetes_Risk",
        hue_order=risk_order,
        palette=risk_palette,
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        kde=True,
        ax=ax,
    )
    ax.set(title="HbA1c distribution by diabetes risk", xlabel="HbA1c", ylabel="Density")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.divider()

    st.markdown("#### Health measurements by risk group")
    for column in ["Blood_Glucose", "Fasting_Blood_Sugar", "HbA1c", "BMI"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=cleaned_data,
            x="Diabetes_Risk",
            y=column,
            order=risk_order,
            hue="Diabetes_Risk",
            palette=risk_palette,
            legend=False,
            ax=ax,
        )
        ax.set(title=column.replace("_", " "), xlabel="Risk group", ylabel=column)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    st.divider()

    st.markdown("#### Age and BMI variation by risk group")
    for column in ["Age", "BMI"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(
            data=cleaned_data, x="Diabetes_Risk", y=column, order=risk_order,
            hue="Diabetes_Risk", palette=risk_palette, legend=False,
            inner="quart", cut=0, ax=ax,
        )
        ax.set(title=f"{column.replace('_', ' ')} distribution", xlabel="Risk group", ylabel=column)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    st.divider()

    st.markdown("#### Relationships between health measurements")
    st.write("Each point represents one record from a reproducible sample of up to 2,500 records.")
    scatter_data = cleaned_data.sample(min(2500, len(cleaned_data)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_data, x="HbA1c", y="Fasting_Blood_Sugar",
        hue="Diabetes_Risk", hue_order=risk_order, palette=risk_palette,
        alpha=0.55, s=28, ax=ax,
    )
    ax.set(title="HbA1c vs fasting blood sugar", xlabel="HbA1c", ylabel="Fasting blood sugar")
    ax.legend(title="Diabetes risk")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_data, x="Age", y="BMI",
        hue="Diabetes_Risk", hue_order=risk_order, palette=risk_palette,
        alpha=0.55, s=28, ax=ax,
    )
    ax.set(title="Age vs BMI", xlabel="Age", ylabel="BMI")
    ax.legend(title="Diabetes risk")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.divider()

    st.markdown("#### Risk mix across lifestyle and health groups")
    group_columns = [
        "Physical_Activity_Level",
        "Diet_Quality",
        "Family_History_Diabetes",
        "Hypertension",
    ]
    group_labels = ["Physical activity", "Diet quality", "Family history", "Hypertension"]
    for column, label in zip(group_columns, group_labels):
        group_risk = pd.crosstab(
            cleaned_data[column], cleaned_data["Diabetes_Risk"], normalize="index"
        ).reindex(columns=risk_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        group_risk.plot(
            kind="bar", stacked=True, ax=ax,
            color=[risk_palette[risk] for risk in risk_order],
            width=0.75,
        )
        ax.set(
            title=f"Risk share by {label.lower()}",
            xlabel=label,
            ylabel="Share of records",
            ylim=(0, 1),
        )
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.legend(title="Risk")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    st.divider()

    st.markdown("#### Correlation between numeric features")
    numeric_data = cleaned_data.select_dtypes(include="number").drop(
        columns=["Patient_ID"], errors="ignore"
    )
    correlation_columns = [
        "Age",
        "BMI",
        "Blood_Glucose",
        "HbA1c",
        "Fasting_Blood_Sugar",
        "Insulin_Level",
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic",
        "Total_Cholesterol",
        "Diabetes_Risk_Score",
    ]
    correlation = numeric_data[correlation_columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation between selected numeric health features")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "Correlation ranges from -1 (strong negative linear association) to +1 "
        "(strong positive linear association); values near 0 indicate little "
        "linear association. Correlation does not establish causation."
    )

with tabs[2]:
    st.subheader("Results, Trends and Conclusions")
    st.write(
        "A concise summary of the cleaned dataset and the patterns visible in "
        "its records. All comparisons below describe this dataset only."
    )
    risk_counts = cleaned_data["Diabetes_Risk"].value_counts()
    risk_percentages = cleaned_data["Diabetes_Risk"].value_counts(normalize=True) * 100
    raw_missing = int(raw_data.isna().sum().sum())
    raw_records_retained = len(cleaned_data) / len(raw_data) if len(raw_data) else 0
    mean_by_risk = cleaned_data.groupby("Diabetes_Risk")[
        ["Age", "BMI", "HbA1c", "Fasting_Blood_Sugar"]
    ].mean().reindex(["Low", "Moderate", "High"])
    high_glucose_means = mean_by_risk.loc["High"]
    low_glucose_means = mean_by_risk.loc["Low"]

    st.markdown("#### Results at a glance")
    result_columns = st.columns(3)
    result_columns[0].metric("Cleaned records", f"{len(cleaned_data):,}")
    result_columns[1].metric("Raw rows retained", f"{raw_records_retained:.1%}")
    result_columns[2].metric("Missing cells after cleaning", f"{int(cleaned_data.isna().sum().sum()):,}")

    st.write(
        f"The raw dataset had **{len(raw_data):,} rows** and **{raw_missing:,} "
        f"missing cells**. The cleaned dataset has **{int(cleaned_data.duplicated().sum()):,} "
        "duplicate rows**."
    )
    risk_results = pd.DataFrame(
        {
            "Records": risk_counts.reindex(["Low", "Moderate", "High"]).fillna(0).astype(int),
            "Share": risk_percentages.reindex(["Low", "Moderate", "High"]).map(
                lambda value: f"{value:.1f}%"
            ),
        }
    )
    st.write("**Risk-group distribution**")
    st.dataframe(risk_results, width="stretch")

    st.divider()
    st.markdown("#### Trends across risk groups")
    st.write(
        "Average age, BMI, HbA1c, and fasting blood sugar increase across the "
        "Low, Moderate, and High risk groups in this dataset."
    )
    st.dataframe(
        mean_by_risk.rename(
            columns={
                "Age": "Mean age",
                "BMI": "Mean BMI",
                "HbA1c": "Mean HbA1c",
                "Fasting_Blood_Sugar": "Mean fasting blood sugar",
            }
        ).round(1),
        width="stretch",
    )

    risk_by_activity = pd.crosstab(
        cleaned_data["Physical_Activity_Level"],
        cleaned_data["Diabetes_Risk"],
        normalize="index",
    )
    risk_by_history = pd.crosstab(
        cleaned_data["Family_History_Diabetes"],
        cleaned_data["Diabetes_Risk"],
        normalize="index",
    )
    st.write(
        f"The high-risk share is **{risk_by_activity.loc['Low', 'High']:.1%}** "
        f"among records with low activity and **{risk_by_activity.loc['High', 'High']:.1%}** "
        f"among records with high activity. It is **{risk_by_history.loc[True, 'High']:.1%}** "
        f"for records with family history and **{risk_by_history.loc[False, 'High']:.1%}** "
        "for those without."
    )

    st.divider()
    st.markdown("#### Key findings")
    score_correlations = (
        cleaned_data.select_dtypes(include="number")
        .drop(columns=["Patient_ID"], errors="ignore")
        .corr()["Diabetes_Risk_Score"]
        .drop("Diabetes_Risk_Score")
    )
    top_correlations = score_correlations.abs().sort_values(ascending=False).head(3)
    correlation_summary = ", ".join(
        f"{column.replace('_', ' ')} (r = {score_correlations[column]:+.2f})"
        for column in top_correlations.index
    )
    st.markdown(
        f"""
        - **Uneven class sizes:** High risk represents {risk_percentages.get('High', 0):.1f}% of records,
          while Low risk represents only {risk_percentages.get('Low', 0):.1f}%. A model should
          therefore be assessed on its performance for each class, not accuracy alone.
        - **Biomarkers differ by label:** High-risk records have mean HbA1c of
          {high_glucose_means['HbA1c']:.1f}, compared with {low_glucose_means['HbA1c']:.1f}
          for Low risk; mean fasting blood sugar is
          {high_glucose_means['Fasting_Blood_Sugar']:.1f} versus
          {low_glucose_means['Fasting_Blood_Sugar']:.1f}.
        - **Largest numeric associations with risk score:** {correlation_summary}.
          These correlations describe linear relationships, not cause and effect.
        """
    )

    st.divider()
    st.markdown("#### Conclusion")
    st.warning(
        f"The cleaned dataset provides {len(cleaned_data):,} complete records for "
        "analysis, but it retains only "
        f"{raw_records_retained:.1%} of the original rows and has a strongly "
        "imbalanced target. Use class-aware metrics when comparing models, and "
        "treat the observed trends as exploratory rather than causal or clinical."
    )