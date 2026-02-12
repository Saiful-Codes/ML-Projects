import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


MODEL_PATH = "models/fraud_pipeline_rf_v1.joblib"
DATA_PATH = "data/raw/creditcard.csv"          # adjust if your path differs
OUT_PATH = "reports/figures/shap_summary.png"


def main():
    # 1) Load model pipeline
    pipeline = joblib.load(MODEL_PATH)

    # 2) Load a sample of data (keep small so it runs fast)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])

    # Use a small sample for speed (e.g., 2000 rows)
    # (RandomForest + SHAP can be slow on full dataset)
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)

    # 3) Transform features the same way the pipeline does
    # This makes sure SHAP explains the *actual* features the model sees.
    # IMPORTANT: adjust the step name if yours differs
    feature_step_name = None
    for name in pipeline.named_steps.keys():
        if "feature" in name.lower():
            feature_step_name = name
            break

    if feature_step_name is None:
        raise ValueError("Could not find the feature engineering step in pipeline.named_steps")

    feature_engineer = pipeline.named_steps[feature_step_name]
    X_fe = feature_engineer.transform(X_sample)

    # Ensure X_fe is a DataFrame with column names (best for readable plots)
    if not isinstance(X_fe, pd.DataFrame):
        # If your transformer returns numpy array, we fall back to generic names
        X_fe = pd.DataFrame(X_fe, columns=[f"feature_{i}" for i in range(X_fe.shape[1])])

    # 4) Get the Random Forest model from pipeline
    model_step_name = None
    for name in pipeline.named_steps.keys():
        if "forest" in name.lower() or "model" in name.lower() or "clf" in name.lower():
            model_step_name = name
    if model_step_name is None:
        # fallback: last step is usually the model
        model_step_name = list(pipeline.named_steps.keys())[-1]

    rf_model = pipeline.named_steps[model_step_name]

    # 5) SHAP explanation for tree-based models
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_fe)

    # For binary classification, shap_values is often [class0, class1]
    # We explain the "fraud" class (usually class 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    # 6) Plot + save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_to_plot, X_fe, show=False)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()
