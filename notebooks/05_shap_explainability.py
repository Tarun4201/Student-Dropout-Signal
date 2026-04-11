# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 05 — SHAP Explainability
# MAGIC ## The Dropout Signal: Global & Per-Student Explanations
# MAGIC
# MAGIC **Input:** `silver.model_test_results` + raw XGBoost from MLflow
# MAGIC
# MAGIC **Outputs:** `shap_summary.png` + `silver.shap_results`

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
# MAGIC ## 5.1 Load Data & Model

# COMMAND ----------

df_test = spark.table("silver.model_test_results").toPandas()
print(f"✅ Test data: {df_test.shape[0]} rows")

NON_FEATURE_COLS = ["target", "dropout_label", "student_id", "dropout_pred", "risk_score"]
FEATURE_COLS = [c for c in df_test.columns if c not in NON_FEATURE_COLS]
X_test = df_test[FEATURE_COLS]
print(f"✅ Features: {len(FEATURE_COLS)}")

# COMMAND ----------

mlflow.set_experiment("/dropout_signal_hackathon")
runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'xgboost_classifier'", order_by=["start_time DESC"])
xgb_run_id = runs.iloc[0].run_id
raw_model = mlflow.sklearn.load_model(f"runs:/{xgb_run_id}/raw_xgb_model")
print(f"✅ Model loaded: {type(raw_model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Compute SHAP Values

# COMMAND ----------

explainer = shap.TreeExplainer(raw_model)
shap_values = explainer.shap_values(X_test)
print(f"✅ SHAP values: {shap_values.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Global Summary Plot

# COMMAND ----------

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance"); plt.tight_layout()
plt.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight"); plt.show()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False, max_display=20)
plt.title("SHAP Beeswarm"); plt.tight_layout()
plt.savefig("/tmp/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.show()

with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact("/tmp/shap_summary.png")
    mlflow.log_artifact("/tmp/shap_beeswarm.png")
print("✅ SHAP plots logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 Top Global Predictors

# COMMAND ----------

mean_abs = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
print("Top 10 Global Predictors:")
for _, row in importance.head(10).iterrows():
    print(f"  {row['feature']:<50s} {row['mean_abs_shap']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 Per-Student Top-3 SHAP Factors

# COMMAND ----------

records = []
for idx in range(len(shap_values)):
    row_shap = shap_values[idx]
    student_id = df_test.iloc[idx]["student_id"]
    top3 = np.argsort(np.abs(row_shap))[::-1][:3]
    records.append({
        "student_id": int(student_id),
        "shap_factor_1": FEATURE_COLS[top3[0]], "shap_value_1": float(row_shap[top3[0]]),
        "shap_factor_2": FEATURE_COLS[top3[1]], "shap_value_2": float(row_shap[top3[1]]),
        "shap_factor_3": FEATURE_COLS[top3[2]], "shap_value_3": float(row_shap[top3[2]]),
    })

shap_df = pd.DataFrame(records)
print(f"✅ Per-student SHAP: {len(shap_df)} students")
print(shap_df.head().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 Write SHAP Results

# COMMAND ----------

shap_spark = spark.createDataFrame(shap_df)
shap_spark.write.format("delta").mode("overwrite").saveAsTable("silver.shap_results")
print(f"✅ silver.shap_results written: {shap_spark.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.7 Validation

# COMMAND ----------

verify = spark.table("silver.shap_results").toPandas()
null_check = verify[["shap_factor_1","shap_factor_2","shap_factor_3",
                       "shap_value_1","shap_value_2","shap_value_3"]].isnull().sum()
for col, n in null_check.items():
    print(f"  {'✅' if n==0 else '❌'} {col}: {n} nulls")

at_risk_ids = set(df_test[df_test["dropout_pred"]==1]["student_id"].astype(int))
shap_ids = set(verify["student_id"].astype(int))
print(f"\nAt-risk: {len(at_risk_ids)} | SHAP coverage: {len(at_risk_ids - shap_ids)} missing")

# COMMAND ----------

# Most common factors for at-risk students
at_risk_shap = verify[verify["student_id"].isin(at_risk_ids)]
factors = at_risk_shap["shap_factor_1"].tolist() + at_risk_shap["shap_factor_2"].tolist() + at_risk_shap["shap_factor_3"].tolist()
print("\nTop SHAP factors for at-risk students:")
for f, c in pd.Series(factors).value_counts().head(10).items():
    print(f"  {f:<50s} {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ SHAP Complete — Next: `06_gold_table`
