import streamlit as st
import pandas as pd

st.title("4) Upload")

uploaded = st.file_uploader("Upload a CSV file to score (later)", type=["csv"])

if uploaded is None:
    st.caption("No file uploaded yet.")
else:
    df = pd.read_csv(uploaded)
    st.success(f"Uploaded file loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

st.info("Day 15: Upload page skeleton only. In Day 17+, we’ll validate columns + run predictions + anomalies.")
