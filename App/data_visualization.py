from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.api.types import is_numeric_dtype


DATA_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "Cleaned_dataset.csv"

st.title("Interactive Data Analysis")
st.write(
    "Choose any two dataset fields and explore how they relate. Available chart "
    "types update automatically to suit the selected fields."
)

if not DATA_PATH.is_file():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

data = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0", "Patient_ID"], errors="ignore")
if data.empty or len(data.columns) < 2:
    st.error("The dataset needs at least two fields to build a visualization.")
    st.stop()


def is_numeric_field(column):
    return is_numeric_dtype(data[column]) and not pd.api.types.is_bool_dtype(data[column])


def field_type(column):
    return "Numeric" if is_numeric_field(column) else "Categorical"


columns = list(data.columns)
numeric_columns = [column for column in columns if is_numeric_field(column)]
if not numeric_columns:
    st.error("No numeric fields are available for visualization.")
    st.stop()

x_column, y_column = st.columns(2)
with x_column:
    x_field = st.selectbox(
        "X-axis",
        options=columns,
        index=columns.index("Age") if "Age" in columns else 0,
        format_func=lambda column: f"{column.replace('_', ' ')} · {field_type(column)}",
    )

y_options = [column for column in columns if column != x_field]
default_y = next(
    (column for column in y_options if column == "Diabetes_Risk"),
    next((column for column in y_options if is_numeric_field(column)), y_options[0]),
)
with y_column:
    y_field = st.selectbox(
        "Y-axis",
        options=y_options,
        index=y_options.index(default_y),
        format_func=lambda column: f"{column.replace('_', ' ')} · {field_type(column)}",
    )

x_numeric = is_numeric_field(x_field)
y_numeric = is_numeric_field(y_field)

if x_numeric and y_numeric:
    chart_options = ["Scatter plot", "Line plot", "Hexbin plot"]
elif not x_numeric and y_numeric:
    chart_options = ["Box plot", "Violin plot", "Bar chart (mean)", "Strip plot"]
elif x_numeric and not y_numeric:
    chart_options = ["Histogram by group", "Density by group", "Box plot", "Violin plot"]
else:
    chart_options = ["Grouped bar chart", "Stacked bar chart", "Count heatmap"]

chart_type = st.selectbox(
    "Chart type",
    options=chart_options,
    help=f"Options are based on X being {field_type(x_field).lower()} and "
    f"Y being {field_type(y_field).lower()}.",
)

chart_data = data[[x_field, y_field]].dropna()
if chart_data.empty:
    st.warning("There are no complete records for this pair of fields.")
    st.stop()

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 6))

if x_numeric and y_numeric:
    if chart_type == "Scatter plot":
        points = chart_data.sample(min(5000, len(chart_data)), random_state=42)
        sns.scatterplot(data=points, x=x_field, y=y_field, alpha=0.55, s=28, ax=ax)
        if len(chart_data) > len(points):
            st.caption("Scatter plot displays a reproducible sample of 5,000 records.")
    elif chart_type == "Line plot":
        sns.lineplot(data=chart_data, x=x_field, y=y_field, ax=ax)
    else:
        sns.histplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            bins=35,
            cbar=True,
            cmap="mako",
            ax=ax,
        )
elif not x_numeric and y_numeric:
    category_order = chart_data[x_field].value_counts().index.tolist()
    if chart_type == "Box plot":
        sns.boxplot(data=chart_data, x=x_field, y=y_field, order=category_order, ax=ax)
    elif chart_type == "Violin plot":
        sns.violinplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            order=category_order,
            inner="quart",
            cut=0,
            ax=ax,
        )
    elif chart_type == "Bar chart (mean)":
        sns.barplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            order=category_order,
            errorbar=("ci", 95),
            ax=ax,
        )
    else:
        sns.stripplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            order=category_order,
            alpha=0.45,
            jitter=0.25,
            ax=ax,
        )
    ax.tick_params(axis="x", rotation=25)
elif x_numeric and not y_numeric:
    category_order = chart_data[y_field].value_counts().index.tolist()
    if chart_type == "Histogram by group":
        sns.histplot(
            data=chart_data,
            x=x_field,
            hue=y_field,
            hue_order=category_order,
            element="step",
            stat="density",
            common_norm=False,
            kde=True,
            ax=ax,
        )
    elif chart_type == "Density by group":
        sns.kdeplot(
            data=chart_data,
            x=x_field,
            hue=y_field,
            hue_order=category_order,
            common_norm=False,
            fill=False,
            ax=ax,
        )
    elif chart_type == "Box plot":
        sns.boxplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            order=category_order,
            orient="h",
            ax=ax,
        )
    else:
        sns.violinplot(
            data=chart_data,
            x=x_field,
            y=y_field,
            order=category_order,
            orient="h",
            inner="quart",
            cut=0,
            ax=ax,
        )
else:
    counts = pd.crosstab(chart_data[x_field], chart_data[y_field])
    if chart_type == "Count heatmap":
        sns.heatmap(counts, annot=len(counts) <= 12 and len(counts.columns) <= 12, fmt="d", cmap="mako", ax=ax)
        ax.set_xlabel(y_field.replace("_", " "))
        ax.set_ylabel(x_field.replace("_", " "))
    else:
        if len(counts) > 20:
            counts = counts.loc[counts.sum(axis=1).nlargest(20).index]
            st.caption("Bar chart is limited to the 20 most common X-axis categories.")
        if len(counts.columns) > 12:
            counts = counts.loc[:, counts.sum(axis=0).nlargest(12).index]
            st.caption("Bar chart is limited to the 12 most common Y-axis categories.")
        counts.plot(
            kind="bar",
            stacked=chart_type == "Stacked bar chart",
            ax=ax,
            colormap="tab20",
        )
        ax.set_xlabel(x_field.replace("_", " "))
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title=y_field.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")

if not (not x_numeric and not y_numeric and chart_type == "Count heatmap"):
    ax.set_xlabel(x_field.replace("_", " "))
    if not (x_numeric and not y_numeric):
        ax.set_ylabel(y_field.replace("_", " "))
ax.set_title(f"{chart_type}: {x_field.replace('_', ' ')} vs {y_field.replace('_', ' ')}")
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

st.caption(
    f"Showing {len(chart_data):,} records with values for both selected fields. "
    "Charts describe patterns in this dataset and do not establish causation."
)