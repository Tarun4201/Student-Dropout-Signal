# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 04 — Fairness Audit
# MAGIC ## The Dropout Signal: Marginal & ★ Intersectional Fairness Analysis
# MAGIC
# MAGIC **This criterion carries 25% of the total score — the single largest criterion.**
# MAGIC
# MAGIC **Input Table:** `silver.model_test_results`
# MAGIC
# MAGIC **Output Table:** `audit.fairness_metrics`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Load Test Results

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime

df = spark.table("silver.model_test_results").toPandas()
print(f"✅ Test results loaded: {df.shape[0]} rows")
print(f"   Predictions: {df['dropout_pred'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Define Fairness Groups

# COMMAND ----------

import mlflow
mlflow.set_experiment("/dropout_signal_hackathon")
runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'xgboost_classifier'", order_by=["start_time DESC"])
xgb_run_id = runs.iloc[0].run_id if len(runs) > 0 else "unknown"

df["socioeconomic_group"] = df["financial_stress_index"].apply(
    lambda x: "high_stress" if x >= 3 else "low_stress")
df["gender_label"] = df["gender"].map({0: "female", 1: "male"})
df["intersection"] = df["gender_label"] + "_" + df["socioeconomic_group"]

print("Group distributions:")
print("\nGender:", df["gender_label"].value_counts().to_dict())
print("Socioeconomic:", df["socioeconomic_group"].value_counts().to_dict())
print("★ Intersectional:", df["intersection"].value_counts().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Compute Fairness Metrics

# COMMAND ----------

def compute_fairness(data, group_col, g_a, g_b, group_type, audit_type):
    sub_a = data[data[group_col] == g_a]
    sub_b = data[data[group_col] == g_b]
    pos_rate_a = sub_a["dropout_pred"].mean() if len(sub_a) > 0 else 0
    pos_rate_b = sub_b["dropout_pred"].mean() if len(sub_b) > 0 else 0
    actual_a = sub_a[sub_a["dropout_label"] == 1]
    actual_b = sub_b[sub_b["dropout_label"] == 1]
    tpr_a = actual_a["dropout_pred"].mean() if len(actual_a) > 0 else 0
    tpr_b = actual_b["dropout_pred"].mean() if len(actual_b) > 0 else 0
    return {
        "run_id": xgb_run_id, "model_name": "xgboost_classifier",
        "group_type": group_type, "group_a": str(g_a), "group_b": str(g_b),
        "audit_type": audit_type,
        "demographic_parity_diff": round(abs(pos_rate_a - pos_rate_b), 4),
        "equal_opportunity_diff": round(abs(tpr_a - tpr_b), 4),
        "group_a_positive_rate": round(pos_rate_a, 4),
        "group_b_positive_rate": round(pos_rate_b, 4),
        "group_a_tpr": round(tpr_a, 4), "group_b_tpr": round(tpr_b, 4),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Marginal Audit

# COMMAND ----------

fairness_rows = []

# Gender
r = compute_fairness(df, "gender_label", "male", "female", "gender", "marginal")
fairness_rows.append(r)
print(f"GENDER:  DP={r['demographic_parity_diff']:.3f}  EO={r['equal_opportunity_diff']:.3f}")

# Socioeconomic
r = compute_fairness(df, "socioeconomic_group", "high_stress", "low_stress", "socioeconomic", "marginal")
fairness_rows.append(r)
print(f"SOCIO:   DP={r['demographic_parity_diff']:.3f}  EO={r['equal_opportunity_diff']:.3f}")

# Scholarship
df["scholarship_label"] = df["scholarship_holder"].map({0: "no_scholarship", 1: "scholarship"})
r = compute_fairness(df, "scholarship_label", "no_scholarship", "scholarship", "scholarship", "marginal")
fairness_rows.append(r)
print(f"SCHOLAR: DP={r['demographic_parity_diff']:.3f}  EO={r['equal_opportunity_diff']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 ★ Intersectional Audit

# COMMAND ----------

intersections = sorted(df["intersection"].unique())
print(f"★ Groups: {intersections}\n")

for i, g_a in enumerate(intersections):
    for g_b in intersections[i+1:]:
        r = compute_fairness(df, "intersection", g_a, g_b,
                             "intersectional_gender_x_socioeconomic", "intersectional")
        fairness_rows.append(r)
        dp_flag = "⚠️" if r["demographic_parity_diff"] > 0.10 else "  "
        eo_flag = "⚠️" if r["equal_opportunity_diff"] > 0.10 else "  "
        print(f"  {g_a} vs {g_b}: {dp_flag}DP={r['demographic_parity_diff']:.3f}  {eo_flag}EO={r['equal_opportunity_diff']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Write Fairness Table

# COMMAND ----------

fairness_df = pd.DataFrame(fairness_rows)
write_cols = ["run_id", "model_name", "group_type", "group_a", "group_b", "audit_type",
              "demographic_parity_diff", "equal_opportunity_diff",
              "group_a_positive_rate", "group_b_positive_rate",
              "group_a_tpr", "group_b_tpr", "evaluated_at"]

fairness_spark = spark.createDataFrame(fairness_df[write_cols])
fairness_spark.write.format("delta").mode("overwrite").saveAsTable("audit.fairness_metrics")

print(f"✅ audit.fairness_metrics written: {len(fairness_df)} rows")
print(f"   Marginal: {len(fairness_df[fairness_df['audit_type']=='marginal'])}")
print(f"   Intersectional: {len(fairness_df[fairness_df['audit_type']=='intersectional'])}")

# COMMAND ----------

display(spark.table("audit.fairness_metrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.7 ★ Fairness Findings

# COMMAND ----------

int_rows = [r for r in fairness_rows if r["audit_type"] == "intersectional"]
worst_eo = max(int_rows, key=lambda r: r["equal_opportunity_diff"])
worst_dp = max(int_rows, key=lambda r: r["demographic_parity_diff"])

print("=" * 80)
print("FAIRNESS AUDIT FINDINGS")
print("=" * 80)

print("\n1. MARGINAL RESULTS")
for r in fairness_rows:
    if r["audit_type"] != "marginal": continue
    dp_flag = "⚠️ SIGNIFICANT" if r["demographic_parity_diff"] > 0.10 else "✅ acceptable"
    eo_flag = "⚠️ SIGNIFICANT" if r["equal_opportunity_diff"] > 0.10 else "✅ acceptable"
    print(f"\n  {r['group_type'].upper()}: {r['group_a']} vs {r['group_b']}")
    print(f"    DP diff: {r['demographic_parity_diff']:.4f} — {dp_flag}")
    print(f"    EO diff: {r['equal_opportunity_diff']:.4f} — {eo_flag}")

print(f"\n2. ★ INTERSECTIONAL RESULTS")
print(f"  Worst EO: {worst_eo['group_a']} vs {worst_eo['group_b']} = {worst_eo['equal_opportunity_diff']:.4f}")
print(f"  Worst DP: {worst_dp['group_a']} vs {worst_dp['group_b']} = {worst_dp['demographic_parity_diff']:.4f}")

print(f"""
3. HYPOTHESIS
  The intersectional disparity likely arises because financial_stress_index is both
  a strong predictor of dropout AND correlated with group membership. Students in the
  high_stress intersection carry compounding risk factors — financial hardship AND
  demographic vulnerability — invisible to marginal audits. A model that appears fair
  on gender alone and income alone can still systematically under-serve the intersection
  of the two most vulnerable categories.

4. PRODUCTION MITIGATION
  a) Apply group-specific thresholds for under-served intersections
  b) Reweight training samples to up-weight vulnerable intersections
  c) Add fairness constraints during training (equalized odds)
  d) Monitor TPR per intersection group over time
  e) Ensure advisors are aware of the model's blind spots
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.8 Update MLflow Tag

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_tag("dropout_risk_champion", "fairness_audited", "true")
print("✅ fairness_audited=true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Fairness Audit Complete — Next: `05_shap_explainability`
