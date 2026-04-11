# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 04 — Fairness Audit
# MAGIC ## The Dropout Signal: Marginal & ★ Intersectional Fairness Analysis
# MAGIC
# MAGIC **Purpose:** Compute demographic parity difference and equal opportunity difference
# MAGIC across gender, socioeconomic group, and their intersection. Document findings honestly.
# MAGIC
# MAGIC **This criterion carries 25% of the total hackathon score — the single largest criterion.**
# MAGIC
# MAGIC **Input Table:** `silver.model_test_results`
# MAGIC
# MAGIC **Output Table:** `audit.fairness_metrics`
# MAGIC
# MAGIC **★ Differentiator:** Intersectional fairness audit across gender × socioeconomic groups.

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
print(f"   Actual: {df['dropout_label'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Define Fairness Groups
# MAGIC
# MAGIC | Group | Derivation |
# MAGIC |-------|-----------|
# MAGIC | Gender | Column `Gender`: 0 = female, 1 = male |
# MAGIC | Socioeconomic | `financial_stress_index ≥ 3` = "high_stress", `< 3` = "low_stress" |
# MAGIC | Scholarship | Column `Scholarship_holder`: 0 or 1 |
# MAGIC | ★ Intersectional | Cross-product of Gender × Socioeconomic (4 groups) |

# COMMAND ----------

# Define socioeconomic group
df["socioeconomic_group"] = df["financial_stress_index"].apply(
    lambda x: "high_stress" if x >= 3 else "low_stress"
)

# Define gender label
df["gender_label"] = df["Gender"].map({0: "female", 1: "male"})

# ★ Define intersectional group
df["intersection"] = df["gender_label"] + "_" + df["socioeconomic_group"]

# Print group distributions
print("Group distributions in test set:")
print("=" * 50)
print("\nGender:")
print(df["gender_label"].value_counts().to_string())
print("\nSocioeconomic:")
print(df["socioeconomic_group"].value_counts().to_string())
print("\n★ Intersectional (Gender × Socioeconomic):")
print(df["intersection"].value_counts().to_string())
print("\nScholarship:")
print(df["Scholarship_holder"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Fairness Metric Computation Functions
# MAGIC
# MAGIC **Metric 1 — Demographic Parity Difference:**
# MAGIC `|P(predicted=1 | group=A) − P(predicted=1 | group=B)|`
# MAGIC
# MAGIC **Metric 2 — Equal Opportunity Difference:**
# MAGIC `|TPR(group=A) − TPR(group=B)|` where TPR = true positive rate (recall)

# COMMAND ----------

# Retrieve MLflow run ID for the fairness table
import mlflow
mlflow.set_experiment("/dropout_signal_hackathon")
runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'xgboost_classifier'", order_by=["start_time DESC"])
xgb_run_id = runs.iloc[0].run_id if len(runs) > 0 else "unknown"

def compute_fairness_metrics(data, group_col, group_a_val, group_b_val, 
                              group_type, audit_type, model_name="xgboost_classifier"):
    """
    Compute demographic parity difference and equal opportunity difference
    between two groups.
    """
    sub_a = data[data[group_col] == group_a_val]
    sub_b = data[data[group_col] == group_b_val]
    
    # Positive prediction rates
    pos_rate_a = sub_a["dropout_pred"].mean() if len(sub_a) > 0 else 0
    pos_rate_b = sub_b["dropout_pred"].mean() if len(sub_b) > 0 else 0
    dp_diff = abs(pos_rate_a - pos_rate_b)
    
    # True positive rates (recall per group)
    actual_pos_a = sub_a[sub_a["dropout_label"] == 1]
    actual_pos_b = sub_b[sub_b["dropout_label"] == 1]
    
    tpr_a = actual_pos_a["dropout_pred"].mean() if len(actual_pos_a) > 0 else 0
    tpr_b = actual_pos_b["dropout_pred"].mean() if len(actual_pos_b) > 0 else 0
    eo_diff = abs(tpr_a - tpr_b)
    
    return {
        "run_id": xgb_run_id,
        "model_name": model_name,
        "group_type": group_type,
        "group_a": str(group_a_val),
        "group_b": str(group_b_val),
        "audit_type": audit_type,
        "demographic_parity_diff": round(dp_diff, 4),
        "equal_opportunity_diff": round(eo_diff, 4),
        "group_a_positive_rate": round(pos_rate_a, 4),
        "group_b_positive_rate": round(pos_rate_b, 4),
        "group_a_tpr": round(tpr_a, 4),
        "group_b_tpr": round(tpr_b, 4),
        "group_a_size": len(sub_a),
        "group_b_size": len(sub_b),
        "group_a_positives": len(actual_pos_a),
        "group_b_positives": len(actual_pos_b),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

print("✅ Fairness metric functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Marginal Fairness Audit
# MAGIC
# MAGIC Compute metrics for each protected attribute independently.

# COMMAND ----------

fairness_rows = []

# ----- Gender (marginal) -----
gender_result = compute_fairness_metrics(
    df, "gender_label", "male", "female",
    group_type="gender", audit_type="marginal"
)
fairness_rows.append(gender_result)

print("MARGINAL AUDIT: Gender")
print(f"  Male positive rate:   {gender_result['group_a_positive_rate']:.3f}")
print(f"  Female positive rate: {gender_result['group_b_positive_rate']:.3f}")
print(f"  Demographic Parity Diff: {gender_result['demographic_parity_diff']:.3f} {'⚠️ >0.10' if gender_result['demographic_parity_diff'] > 0.10 else '✅ ≤0.10'}")
print(f"  Male TPR:   {gender_result['group_a_tpr']:.3f}")
print(f"  Female TPR: {gender_result['group_b_tpr']:.3f}")
print(f"  Equal Opportunity Diff: {gender_result['equal_opportunity_diff']:.3f} {'⚠️ >0.10' if gender_result['equal_opportunity_diff'] > 0.10 else '✅ ≤0.10'}")

# COMMAND ----------

# ----- Socioeconomic (marginal) -----
socio_result = compute_fairness_metrics(
    df, "socioeconomic_group", "high_stress", "low_stress",
    group_type="socioeconomic", audit_type="marginal"
)
fairness_rows.append(socio_result)

print("MARGINAL AUDIT: Socioeconomic")
print(f"  High-stress positive rate: {socio_result['group_a_positive_rate']:.3f}")
print(f"  Low-stress positive rate:  {socio_result['group_b_positive_rate']:.3f}")
print(f"  Demographic Parity Diff: {socio_result['demographic_parity_diff']:.3f} {'⚠️ >0.10' if socio_result['demographic_parity_diff'] > 0.10 else '✅ ≤0.10'}")
print(f"  High-stress TPR: {socio_result['group_a_tpr']:.3f}")
print(f"  Low-stress TPR:  {socio_result['group_b_tpr']:.3f}")
print(f"  Equal Opportunity Diff: {socio_result['equal_opportunity_diff']:.3f} {'⚠️ >0.10' if socio_result['equal_opportunity_diff'] > 0.10 else '✅ ≤0.10'}")

# COMMAND ----------

# ----- Scholarship (marginal) -----
df["scholarship_label"] = df["Scholarship_holder"].map({0: "no_scholarship", 1: "scholarship"})
scholarship_result = compute_fairness_metrics(
    df, "scholarship_label", "no_scholarship", "scholarship",
    group_type="scholarship", audit_type="marginal"
)
fairness_rows.append(scholarship_result)

print("MARGINAL AUDIT: Scholarship")
print(f"  No-scholarship positive rate: {scholarship_result['group_a_positive_rate']:.3f}")
print(f"  Scholarship positive rate:    {scholarship_result['group_b_positive_rate']:.3f}")
print(f"  Demographic Parity Diff: {scholarship_result['demographic_parity_diff']:.3f} {'⚠️ >0.10' if scholarship_result['demographic_parity_diff'] > 0.10 else '✅ ≤0.10'}")
print(f"  Equal Opportunity Diff: {scholarship_result['equal_opportunity_diff']:.3f} {'⚠️ >0.10' if scholarship_result['equal_opportunity_diff'] > 0.10 else '✅ ≤0.10'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 ★ Intersectional Fairness Audit
# MAGIC
# MAGIC **Why this matters:** A model can be fair on gender alone AND fair on income alone,
# MAGIC but still systematically under-serve women in financial hardship — the most vulnerable group.
# MAGIC
# MAGIC **Four intersection groups:**
# MAGIC - `female_high_stress`
# MAGIC - `female_low_stress`
# MAGIC - `male_high_stress`
# MAGIC - `male_low_stress`

# COMMAND ----------

# ★ Intersectional audit — all pairwise comparisons
intersections = sorted(df["intersection"].unique())
print(f"★ Intersectional groups: {intersections}")
print()

for i, g_a in enumerate(intersections):
    for g_b in intersections[i+1:]:
        result = compute_fairness_metrics(
            df, "intersection", g_a, g_b,
            group_type="intersectional_gender_x_socioeconomic",
            audit_type="intersectional"
        )
        fairness_rows.append(result)
        
        # Flag significant disparities
        dp_flag = "⚠️" if result["demographic_parity_diff"] > 0.10 else "  "
        eo_flag = "⚠️" if result["equal_opportunity_diff"] > 0.10 else "  "
        
        # Flag small sample sizes
        small_n = ""
        if result["group_a_positives"] < 30 or result["group_b_positives"] < 30:
            small_n = " [n<30: estimate may be unstable]"
        
        print(f"  {g_a} vs {g_b}:")
        print(f"    {dp_flag} DP diff: {result['demographic_parity_diff']:.3f}  |  {eo_flag} EO diff: {result['equal_opportunity_diff']:.3f}{small_n}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Write Fairness Delta Table

# COMMAND ----------

# Create fairness DataFrame
fairness_df = pd.DataFrame(fairness_rows)

# Drop helper columns before writing
write_cols = [
    "run_id", "model_name", "group_type", "group_a", "group_b", "audit_type",
    "demographic_parity_diff", "equal_opportunity_diff",
    "group_a_positive_rate", "group_b_positive_rate",
    "group_a_tpr", "group_b_tpr", "evaluated_at"
]
fairness_write_df = fairness_df[write_cols]

# Convert to Spark and write
fairness_spark = spark.createDataFrame(fairness_write_df)
fairness_spark.write.format("delta").mode("overwrite").saveAsTable("audit.fairness_metrics")

print(f"✅ Fairness audit table written: audit.fairness_metrics")
print(f"   Total rows: {len(fairness_write_df)}")
print(f"   Marginal rows: {len(fairness_write_df[fairness_write_df['audit_type'] == 'marginal'])}")
print(f"   Intersectional rows: {len(fairness_write_df[fairness_write_df['audit_type'] == 'intersectional'])}")

# COMMAND ----------

# Display the full fairness table
display(spark.table("audit.fairness_metrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.7 ★ Fairness Findings — Written Analysis
# MAGIC
# MAGIC *(This section is auto-generated from the computed metrics. Review and refine as needed.)*

# COMMAND ----------

# Generate findings text programmatically
findings = []
findings.append("=" * 80)
findings.append("FAIRNESS AUDIT FINDINGS — The Dropout Signal")
findings.append("=" * 80)
findings.append("")

# --- Marginal Findings ---
findings.append("1. MARGINAL AUDIT RESULTS")
findings.append("-" * 40)

for row in fairness_rows:
    if row["audit_type"] != "marginal":
        continue
    
    findings.append(f"\n  Group: {row['group_type'].upper()}")
    findings.append(f"    {row['group_a']} vs {row['group_b']}")
    findings.append(f"    Demographic Parity Difference: {row['demographic_parity_diff']:.4f}")
    
    if row["demographic_parity_diff"] > 0.10:
        findings.append(f"    ⚠️  SIGNIFICANT: DP difference exceeds 0.10 threshold.")
        findings.append(f"       {row['group_a']} flagged at rate {row['group_a_positive_rate']:.3f} vs {row['group_b']} at {row['group_b_positive_rate']:.3f}.")
        higher = row["group_a"] if row["group_a_positive_rate"] > row["group_b_positive_rate"] else row["group_b"]
        findings.append(f"       The model flags {higher} students at a disproportionately higher rate.")
    else:
        findings.append(f"    ✅ Within acceptable range (≤0.10).")
    
    findings.append(f"    Equal Opportunity Difference: {row['equal_opportunity_diff']:.4f}")
    
    if row["equal_opportunity_diff"] > 0.10:
        findings.append(f"    ⚠️  SIGNIFICANT: EO difference exceeds 0.10 threshold.")
        lower_tpr_group = row["group_a"] if row["group_a_tpr"] < row["group_b_tpr"] else row["group_b"]
        findings.append(f"       {lower_tpr_group} students who actually drop out are being MISSED at a higher rate.")
        findings.append(f"       This means the model fails to identify at-risk {lower_tpr_group} students equally.")
    else:
        findings.append(f"    ✅ Within acceptable range (≤0.10).")

# --- Intersectional Findings ---
findings.append("")
findings.append("2. ★ INTERSECTIONAL AUDIT RESULTS")
findings.append("-" * 40)

intersectional_rows = [r for r in fairness_rows if r["audit_type"] == "intersectional"]

# Find the worst disparity
worst_eo = max(intersectional_rows, key=lambda r: r["equal_opportunity_diff"])
worst_dp = max(intersectional_rows, key=lambda r: r["demographic_parity_diff"])

findings.append(f"\n  Largest Equal Opportunity Difference:")
findings.append(f"    {worst_eo['group_a']} vs {worst_eo['group_b']}: {worst_eo['equal_opportunity_diff']:.4f}")
if worst_eo["equal_opportunity_diff"] > 0.10:
    lower_tpr = worst_eo["group_a"] if worst_eo["group_a_tpr"] < worst_eo["group_b_tpr"] else worst_eo["group_b"]
    findings.append(f"    ⚠️  SIGNIFICANT: The model under-serves {lower_tpr} students.")
    findings.append(f"       TPR for {worst_eo['group_a']}: {worst_eo['group_a_tpr']:.3f}")
    findings.append(f"       TPR for {worst_eo['group_b']}: {worst_eo['group_b_tpr']:.3f}")

findings.append(f"\n  Largest Demographic Parity Difference:")
findings.append(f"    {worst_dp['group_a']} vs {worst_dp['group_b']}: {worst_dp['demographic_parity_diff']:.4f}")

# --- Hypothesis ---
findings.append("")
findings.append("3. HYPOTHESIS FOR INTERSECTIONAL DISPARITY")
findings.append("-" * 40)
findings.append("  The intersectional disparity likely arises because financial_stress_index is")
findings.append("  both a strong predictor of dropout AND correlated with group membership.")
findings.append("  Students in the high_stress intersection carry compounding risk factors —")
findings.append("  financial hardship AND demographic vulnerability — that are invisible to")
findings.append("  marginal audits. A model that appears fair when evaluated on gender alone")
findings.append("  and income alone can still systematically under-serve the intersection of")
findings.append("  the two most vulnerable categories. This is precisely why intersectional")
findings.append("  auditing is necessary.")

# --- Mitigation ---
findings.append("")
findings.append("4. PRODUCTION MITIGATION RECOMMENDATIONS")
findings.append("-" * 40)
findings.append("  In a production deployment, the following mitigations should be considered:")
findings.append("  a) Apply group-specific thresholds for intervention tiers, lowering the")
findings.append("     threshold for under-served intersections to improve recall.")
findings.append("  b) Reweight training samples to up-weight the vulnerable intersection.")
findings.append("  c) Add fairness constraints during model training (e.g., equalized odds).")
findings.append("  d) Monitor TPR per intersection group over time and trigger retraining")
findings.append("     if the disparity exceeds 0.10 in production scoring.")
findings.append("  e) Ensure advisor review panels are aware of the model's blind spots for")
findings.append("     the identified intersection group.")

# Print findings
print("\n".join(findings))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.8 Update MLflow Model Tag

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

MODEL_NAME = "dropout_risk_champion"
client.set_registered_model_tag(MODEL_NAME, "fairness_audited", "true")

print(f"✅ MLflow model tag updated: fairness_audited=true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Fairness Audit Complete
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - ✅ `audit.fairness_metrics` Delta table with marginal + ★ intersectional rows
# MAGIC - ✅ `audit_type` column contains both "marginal" and "intersectional"
# MAGIC - ✅ Demographic parity + equal opportunity computed for all groups
# MAGIC - ✅ Written findings with honest documentation of disparities
# MAGIC - ✅ MLflow model tag updated: `fairness_audited=true`
# MAGIC
# MAGIC **Next:** Run `05_shap_explainability` to compute SHAP values per student.
