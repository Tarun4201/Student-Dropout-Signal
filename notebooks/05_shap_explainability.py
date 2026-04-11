# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 05 — SHAP Explainability
# MAGIC ## The Dropout Signal: Global & Per-Student Explanations
# MAGIC
# MAGIC **Purpose:** Run SHAP TreeExplainer on the XGBoost model to surface the top-3
# MAGIC risk factors per flagged student. Generate the global summary plot.
# MAGIC
# MAGIC **Input:** `silver.model_test_results` + raw XGBoost model from MLflow
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `shap_summary.png` logged to MLflow
# MAGIC - `silver.shap_results` Delta table with per-student top-3 factors

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Imports & Setup

# COMMAND ----------

import shap
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print(f"✅ SHAP version: {shap.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Load Test Data & Model

# COMMAND ----------

# Load test results from the Silver table
df_test = spark.table("silver.model_test_results").toPandas()
print(f"✅ Test data loaded: {df_test.shape[0]} rows")

# COMMAND ----------

# Identify feature columns (exclude target, predictions, metadata)
NON_FEATURE_COLS = ["Target", "dropout_label", "student_id", "dropout_pred", "risk_score"]
FEATURE_COLS = [c for c in df_test.columns if c not in NON_FEATURE_COLS]

X_test = df_test[FEATURE_COLS]
print(f"✅ Feature columns identified: {len(FEATURE_COLS)}")

# COMMAND ----------

# Load the raw XGBoost model from MLflow (not the calibrated wrapper)
mlflow.set_experiment("/dropout_signal_hackathon")
runs = mlflow.search_runs(
    filter_string="tags.mlflow.runName = 'xgboost_classifier'",
    order_by=["start_time DESC"]
)
xgb_run_id = runs.iloc[0].run_id
print(f"✅ XGBoost run found: {xgb_run_id}")

# Load raw XGBoost model
raw_model = mlflow.sklearn.load_model(f"runs:/{xgb_run_id}/raw_xgb_model")
print(f"✅ Raw XGBoost model loaded: {type(raw_model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Compute SHAP Values
# MAGIC
# MAGIC Using `TreeExplainer` — exact and fast for tree-based models.
# MAGIC Computing on the test set only to avoid data leakage in explanations.

# COMMAND ----------

# Create TreeExplainer (fast, exact for tree models)
explainer = shap.TreeExplainer(raw_model)

# Compute SHAP values on the test set
shap_values = explainer.shap_values(X_test)

print(f"✅ SHAP values computed")
print(f"   Shape: {shap_values.shape}")
print(f"   Students: {shap_values.shape[0]}")
print(f"   Features: {shap_values.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 Global Explainability — Summary Plot
# MAGIC
# MAGIC Bar chart of mean absolute SHAP values across all features.
# MAGIC This shows which features are most important globally.

# COMMAND ----------

# Generate SHAP summary plot (bar chart)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ SHAP summary bar chart saved")

# COMMAND ----------

# Also generate the detailed beeswarm plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False, max_display=20)
plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ SHAP beeswarm plot saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 Log SHAP Artifacts to MLflow

# COMMAND ----------

with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact("/tmp/shap_summary.png")
    mlflow.log_artifact("/tmp/shap_beeswarm.png")

print("✅ SHAP plots logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 Top-5 Global Predictors
# MAGIC
# MAGIC These features have the highest mean absolute SHAP values and should appear in the presentation.

# COMMAND ----------

# Identify top global predictors
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("Top 10 Global Predictors (by mean |SHAP|):")
print("=" * 50)
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:<45s} {row['mean_abs_shap']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.7 Per-Student Top-3 SHAP Factors
# MAGIC
# MAGIC For each student in the test set, extract the three features with the highest
# MAGIC absolute SHAP values. These drive the `reason_text` generation in Notebook 06.

# COMMAND ----------

feature_names = FEATURE_COLS

records = []
for idx in range(len(shap_values)):
    row_shap = shap_values[idx]
    student_id = df_test.iloc[idx]["student_id"]
    
    # Get top 3 features by absolute SHAP value
    top3_idx = np.argsort(np.abs(row_shap))[::-1][:3]
    
    record = {
        "student_id": int(student_id),
        "shap_factor_1": feature_names[top3_idx[0]],
        "shap_value_1": float(row_shap[top3_idx[0]]),
        "shap_factor_2": feature_names[top3_idx[1]],
        "shap_value_2": float(row_shap[top3_idx[1]]),
        "shap_factor_3": feature_names[top3_idx[2]],
        "shap_value_3": float(row_shap[top3_idx[2]]),
    }
    records.append(record)

shap_df = pd.DataFrame(records)
print(f"✅ Per-student SHAP factors extracted: {len(shap_df)} students")

# Preview
print("\nSample (first 5 students):")
print(shap_df.head().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.8 Write SHAP Results Delta Table

# COMMAND ----------

shap_spark = spark.createDataFrame(shap_df)
shap_spark.write.format("delta").mode("overwrite").saveAsTable("silver.shap_results")

print(f"✅ SHAP results written to: silver.shap_results")
print(f"   Rows: {shap_spark.count()}")
print(f"   Columns: {len(shap_spark.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.9 Validation

# COMMAND ----------

# Verify no nulls in SHAP results
verify = spark.table("silver.shap_results").toPandas()

null_check = verify[["shap_factor_1", "shap_factor_2", "shap_factor_3",
                       "shap_value_1", "shap_value_2", "shap_value_3"]].isnull().sum()

print("Null check on SHAP results:")
for col, nulls in null_check.items():
    print(f"  {'✅' if nulls == 0 else '❌'} {col}: {nulls} nulls")

# Check that at-risk students all have SHAP values
at_risk = df_test[df_test["dropout_pred"] == 1]
at_risk_ids = set(at_risk["student_id"].astype(int))
shap_ids = set(verify["student_id"].astype(int))
missing = at_risk_ids - shap_ids

print(f"\nAt-risk students: {len(at_risk_ids)}")
print(f"Students with SHAP: {len(shap_ids)}")
print(f"Missing SHAP for at-risk: {len(missing)} {'✅' if len(missing) == 0 else '❌'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.10 Most Common Top SHAP Factors for At-Risk Students

# COMMAND ----------

# Which features most commonly appear as top factors for at-risk students?
at_risk_shap = verify[verify["student_id"].isin(at_risk_ids)]
all_factors = (
    at_risk_shap["shap_factor_1"].tolist() +
    at_risk_shap["shap_factor_2"].tolist() +
    at_risk_shap["shap_factor_3"].tolist()
)
factor_counts = pd.Series(all_factors).value_counts()

print("Most common SHAP factors for at-risk students:")
print("=" * 50)
for feature, count in factor_counts.head(10).items():
    pct = count / len(at_risk_shap) * 100
    print(f"  {feature:<45s} {count:4d} appearances ({pct:.0f}% of at-risk students)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ SHAP Explainability Complete
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - ✅ `shap_summary.png` logged to MLflow (global feature importance)
# MAGIC - ✅ `shap_beeswarm.png` logged to MLflow (detailed impact plot)
# MAGIC - ✅ Per-student top-3 factors saved to `silver.shap_results`
# MAGIC - ✅ All at-risk students have non-null SHAP values
# MAGIC - ✅ TreeExplainer used (not KernelExplainer)
# MAGIC
# MAGIC **Next:** Run `06_gold_table` to build the final Gold output with reason_text.
