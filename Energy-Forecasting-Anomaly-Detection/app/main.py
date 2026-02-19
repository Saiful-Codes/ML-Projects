import streamlit as st
from utils import load_model, load_threshold_config, load_default_data

st.set_page_config(
    page_title="Energy Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting & Anomaly Detection")
st.caption("Week 3: Streamlit App — Day 15 (Skeleton)")

# Sidebar: quick status checks
st.sidebar.header("System Status")

model, model_err = load_model()
cfg, cfg_err = load_threshold_config()
df, df_err = load_default_data()

if model_err:
    st.sidebar.error(model_err)
else:
    st.sidebar.success("Model loaded")

if cfg_err:
    st.sidebar.error(cfg_err)
else:
    st.sidebar.success("Threshold config loaded")

if df_err:
    st.sidebar.error(df_err)
else:
    st.sidebar.success(f"Default data loaded ({df.shape[0]:,} rows, {df.shape[1]} cols)")

st.markdown("""
### What you can do in this app
Use the pages on the left:
- **Overview**: dataset + quick health checks
- **Forecast**: generate predictions (coming next)
- **Anomalies**: view flagged residual anomalies (coming next)
- **Upload**: upload your own CSV for scoring (coming next)
""")

st.info("Day 15 goal: make sure the multi-page skeleton runs smoothly. Real forecasting + anomaly logic comes in Day 16+.")
