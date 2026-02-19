import streamlit as st
from utils import load_threshold_config, load_default_data

st.title("3) Anomalies")

cfg, cfg_err = load_threshold_config()
df, df_err = load_default_data()

if cfg_err:
    st.error(cfg_err)
    st.stop()
if df_err:
    st.error(df_err)
    st.stop()

st.info("Anomaly scoring + charts will be implemented next. Today we confirm config loads correctly.")
st.write("Threshold config:")
st.json(cfg)
st.write("Data shape:", df.shape)
