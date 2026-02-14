import sys
import os

# Add project root to Python path (so joblib can import src.* modules inside the saved pipeline)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


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
    required_cols = getattr(pipeline, "feature_names_in_", None)
    if required_cols is None:
        # If not available, we can't strictly validate here.
        return []
    required_cols = list(required_cols)
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def set_plot_defaults():
    """
    Recommended compact plot settings:
    - Smaller default figure size (so you don't need to scroll)
    - Slightly smaller font size
    """
    plt.rcParams["figure.figsize"] = (9, 4)
    plt.rcParams["font.size"] = 10


def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    st.title("Fraud Detection System")
    st.write("Upload a CSV file and run the trained model to get fraud risk predictions.")

    # Apply compact plotting defaults for the whole app
    set_plot_defaults()

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

    # Optional: if user uploads the full dataset including target
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

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

            # --- Metrics ---
            total_rows = len(results)
            flagged_count = int(results["fraud_flag"].sum())
            flagged_rate = (flagged_count / total_rows) * 100 if total_rows > 0 else 0.0

            st.success("Prediction complete.")

            st.subheader("Summary Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total transactions", f"{total_rows}")
            c2.metric("Flagged as fraud", f"{flagged_count}")
            c3.metric("Flagged rate", f"{flagged_rate:.2f}%")
            c4.metric("Threshold used", f"{threshold:.5f}")

            # --- Charts (compact) ---
            st.subheader("Risk Distribution")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(results["fraud_probability"], bins=30)
            ax.set_xlabel("Fraud probability")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

            st.subheader("Flagged vs Not Flagged")
            counts = results["fraud_flag"].value_counts().sort_index()
            labels = ["Not Flagged (0)", "Flagged (1)"]

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
            ax2.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=False)

            # --- Tables ---
            st.subheader("Output: Full Results")
            st.dataframe(results.head(20), use_container_width=True)

            st.subheader("Top 20 Most Risky Transactions")
            top20 = results.sort_values("fraud_probability", ascending=False).head(20)
            st.dataframe(top20, use_container_width=True)

            # --- Download button ---
            st.subheader("Download Results")
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

            st.caption(f"Using saved threshold: {threshold:.5f}")

        except Exception as e:
            st.error("Model failed to run on this file.")
            st.write("Common reasons: wrong column names, wrong data types, or extra missing values.")
            st.write(f"Error: {e}")


if __name__ == "__main__":
    main()
