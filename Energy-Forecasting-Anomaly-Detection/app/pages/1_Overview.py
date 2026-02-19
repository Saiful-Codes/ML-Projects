import streamlit as st
from utils import load_default_data

st.title("1) Overview")

df, err = load_default_data()
if err:
    st.error(err)
    st.stop()

st.write("Quick look at the processed hourly dataset:")
st.dataframe(df.head(20), use_container_width=True)

st.write("Basic summary:")
st.write(df.describe(include="all"))
