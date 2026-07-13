# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DGA Dashboard", layout="wide")
st.title("Transformer DGA Monitoring Dashboard")

@st.cache_data
def load_data():
    df = pd.read_parquet("dataset/processed/dga_after_severity.parquet")
    ranking = pd.read_csv("dataset/processed/transformer_ranking.csv")
    return df, ranking

df, ranking = load_data()

# Sidebar filters
transformer = st.sidebar.selectbox("Select Transformer", df["transformer_id"].unique())
st.sidebar.markdown("## Top 10 Critical Transformers")
st.sidebar.dataframe(ranking.head(10)[["rank", "transformer_id", "final_score"]])

# Main panel
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Gas Trends for {transformer}")
    tx_df = df[df["transformer_id"] == transformer]
    fig = px.line(tx_df, x="sample_day", y=["h2", "ch4", "c2h2", "c2h4", "co"],
                  title="Key Gases Over Time")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Duval Triangle (latest)")
    latest = tx_df.iloc[-1]
    # Hiển thị tam giác (có thể dùng plotly ternary)
    fig2 = px.scatter_ternary(tx_df, a="pct_ch4", b="pct_c2h4", c="pct_c2h2",
                               color="duval_triangle_fault")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Current Fault & Severity")
st.write(f"Consensus Fault: {latest['consensus_fault']}, Severity Level: {latest['severity_level']}")

# Bảng ranking
st.subheader("Transformer Ranking")
st.dataframe(ranking.style.background_gradient(subset=["final_score"], cmap="Reds"))