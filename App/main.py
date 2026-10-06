import streamlit as st

st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: flex;
    flex-direction: column;
}

[data-testid="stSidebarNav"]::before {
    content: "🩺 Diabetes Risk Prediction System";
    font-size: 24px;
    font-weight: bold;
    padding: 20px 10px;
    order: -1;
}
</style>
""", unsafe_allow_html=True)


dashboard = st.Page('Dashboard.py', title='📈Dashboard')
form = st.Page('form.py', title='📄Form')
eda_report = st.Page('EDA.py', title='📊EDA Report')
models = st.Page('models.py', title='⚙️Models')
analysis = st.Page('data_visualization.py', title='🔍Data Analysis')

pg = st.navigation({
    '🏠Home': [dashboard, eda_report, models, analysis],
    'Prediction Form': [form]
})

pg.run()