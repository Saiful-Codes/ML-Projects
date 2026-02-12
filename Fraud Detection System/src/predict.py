import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Inference Script")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--model", default="models/fraud_pipeline_rf_v1.joblib", help="Path to saved pipeline joblib")
    parser.add_argument("--threshold", default="models/fraud_threshold_v1.joblib", help="Path to saved threshold joblib")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    threshold_path = Path(args.threshold)

    # Load input
    df = pd.read_csv(input_path)

    # Drop target column if present (in case input CSV includes labels)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Load pipeline + threshold
    pipeline = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    # Predict probabilities and labels
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Output results
    out_df = df.copy()
    out_df["fraud_probability"] = probs
    out_df["fraud_prediction"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Threshold used: {threshold}")
    print(f"Predicted fraud count: {preds.sum()} / {len(preds)}")


if __name__ == "__main__":
    main()