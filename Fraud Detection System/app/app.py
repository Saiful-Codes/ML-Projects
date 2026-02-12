import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/fraud_pipeline_rf_v1.joblib"
THRESHOLD_PATH = "models/fraud_threshold_v1.joblib"


@st.cache_resource
def load_assets():
    pipeline = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return pipeline, float(threshold)


def validate_input(df: pd.DataFrame, pipeline) -> list:
    """
    Returns a list of missing required columns.
    We infer required columns from what the pipeline expects at predict time.
    For your project, it should match the raw feature columns like:
    Time, V1..V28, Amount
    """
    # If your pipeline was trained on these exact columns, it will expect them.
    # We'll use a safe check: attempt a small transform/predict later too.
    required_cols = getattr(pipeline, "feature_names_in_", None)
    if required_cols is None:
        # If not available, we can't strictly validate here.
        return []
    required_cols = list(required_cols)
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    st.title("Fraud Detection System")
    st.write("Upload a CSV file and run the trained model to get fraud risk predictions.")

    pipeline, threshold = load_assets()

    st.subheader("1) Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to continue.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file as CSV. Error: {e}")
        return

    st.write("Preview of uploaded data:")
    st.dataframe(df.head(10), use_container_width=True)

    # Validate columns (basic)
    missing_cols = validate_input(df, pipeline)
    if missing_cols:
        st.error("Your CSV is missing required columns:")
        st.write(missing_cols)
        st.stop()

    st.subheader("2) Run Model")
    if st.button("Run Prediction"):
        try:
            # Predict probabilities for Class=1 (fraud)
            proba = pipeline.predict_proba(df)[:, 1]

            results = df.copy()
            results["fraud_probability"] = proba
            results["fraud_flag"] = (results["fraud_probability"] >= threshold).astype(int)

            st.success("Prediction complete.")

            st.subheader("Output: Full Results")
            st.dataframe(results.head(20), use_container_width=True)

            st.subheader("Top 20 Most Risky Transactions")
            top20 = results.sort_values("fraud_probability", ascending=False).head(20)
            st.dataframe(top20, use_container_width=True)

            st.caption(f"Using saved threshold: {threshold:.5f}")

        except Exception as e:
            st.error("Model failed to run on this file.")
            st.write("Common reasons: wrong column names, wrong data types, or extra missing values.")
            st.write(f"Error: {e}")


if __name__ == "__main__":
    main()
