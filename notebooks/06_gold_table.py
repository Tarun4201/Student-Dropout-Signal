# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 06 — Gold Output Table
# MAGIC ## The Dropout Signal: Final At-Risk Students with ★ reason_text
# MAGIC
# MAGIC **Input:** `silver.model_test_results` + `silver.shap_results`
# MAGIC
# MAGIC **Output:** `gold.at_risk_students`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Load Data

# COMMAND ----------

import pandas as pd
from datetime import datetime

test_df = spark.table("silver.model_test_results").toPandas()
shap_df = spark.table("silver.shap_results").toPandas()
print(f"✅ Test results: {test_df.shape[0]} | SHAP: {shap_df.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Join & Filter At-Risk

# COMMAND ----------

test_df["student_id"] = test_df["student_id"].astype(int)
shap_df["student_id"] = shap_df["student_id"].astype(int)

gold_df = test_df.merge(shap_df, on="student_id", how="inner")
gold_df = gold_df[gold_df["dropout_pred"] == 1].copy()
print(f"✅ At-risk students: {len(gold_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Intervention Tiers
# MAGIC
# MAGIC | Tier | Rule | Action |
# MAGIC |------|------|--------|
# MAGIC | HIGH | ≥0.70 | Immediate outreach |
# MAGIC | MEDIUM | 0.40–0.70 | Check-in within 2 weeks |
# MAGIC | LOW | <0.40 | Monitor only |

# COMMAND ----------

gold_df["intervention_tier"] = gold_df["risk_score"].apply(
    lambda s: "high" if s >= 0.70 else ("medium" if s >= 0.40 else "low"))

tier_dist = gold_df["intervention_tier"].value_counts()
for t in ["high", "medium", "low"]:
    c = tier_dist.get(t, 0)
    print(f"  {t.upper():>8}: {c} ({c/len(gold_df)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 ★ Generate `reason_text`
# MAGIC
# MAGIC Plain-English sentences from top-3 SHAP factors for advisors.

# COMMAND ----------

factor_interpretations = {
    "grade_delta": lambda v: f"grade {'fell' if v < 0 else 'rose'} {abs(v):.1f}pts semester-on-semester",
    "financial_stress_index": lambda v: f"financial stress score {v:.0f}/5",
    "absenteeism_trend": lambda v: f"{v*100:.0f}% unit non-completion rate",
    "engagement_score": lambda v: f"engagement score of {v:.2f}",
    "curricular_units_2nd_sem_grade": lambda v: f"semester 2 grade of {v:.1f}",
    "curricular_units_1st_sem_grade": lambda v: f"semester 1 grade of {v:.1f}",
    "curricular_units_2nd_sem_approved": lambda v: f"{int(v)} units approved in semester 2",
    "curricular_units_1st_sem_approved": lambda v: f"{int(v)} units approved in semester 1",
    "curricular_units_2nd_sem_enrolled": lambda v: f"{int(v)} units enrolled in semester 2",
    "curricular_units_1st_sem_enrolled": lambda v: f"{int(v)} units enrolled in semester 1",
    "curricular_units_2nd_sem_evaluations": lambda v: f"{int(v)} evaluations in semester 2",
    "curricular_units_1st_sem_evaluations": lambda v: f"{int(v)} evaluations in semester 1",
    "debtor": lambda v: "outstanding debt on record" if v == 1 else "no debt",
    "tuition_fees_up_to_date": lambda v: "tuition fees overdue" if v == 0 else "fees current",
    "scholarship_holder": lambda v: "no scholarship" if v == 0 else "scholarship holder",
    "age_at_enrollment": lambda v: f"enrolled at age {int(v)}",
    "admission_grade": lambda v: f"admission grade of {v:.1f}",
    "previous_qualification_grade": lambda v: f"previous qualification grade of {v:.1f}",
    "unemployment_rate": lambda v: f"unemployment rate of {v:.1f}%",
    "gdp": lambda v: f"GDP growth of {v:.2f}%",
    "inflation_rate": lambda v: f"inflation rate of {v:.1f}%",
    "displaced": lambda v: "displaced student" if v == 1 else "non-displaced student",
    "international": lambda v: "international student" if v == 1 else "domestic student",
    "gender": lambda v: "male" if v == 1 else "female",
    "daytime_evening_attendance": lambda v: "daytime attendance" if v == 1 else "evening attendance",
    "curricular_units_1st_sem_credited": lambda v: f"{int(v)} units credited in sem 1",
    "curricular_units_2nd_sem_credited": lambda v: f"{int(v)} units credited in sem 2",
    "curricular_units_1st_sem_without_evaluations": lambda v: f"{int(v)} units without eval in sem 1",
    "curricular_units_2nd_sem_without_evaluations": lambda v: f"{int(v)} units without eval in sem 2",
}

def build_reason_text(row):
    reasons = []
    for i in range(1, 4):
        feature = row[f"shap_factor_{i}"]
        value = row.get(feature, None)
        if feature in factor_interpretations and value is not None:
            try:
                reasons.append(factor_interpretations[feature](value))
            except (ValueError, TypeError):
                reasons.append(feature.replace("_", " "))
        else:
            reasons.append(feature.replace("_", " "))
    return "; ".join(reasons).capitalize() + "."

gold_df["reason_text"] = gold_df.apply(build_reason_text, axis=1)
print(f"✅ reason_text generated for {len(gold_df)} students")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ★ Sample reason_text

# COMMAND ----------

for _, row in gold_df.sort_values("risk_score", ascending=False).head(10).iterrows():
    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[row["intervention_tier"]]
    print(f"\n  Student {int(row['student_id'])} | Risk: {row['risk_score']:.3f} | {emoji} {row['intervention_tier'].upper()}")
    print(f"  → {row['reason_text']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Build Final Gold Schema

# COMMAND ----------

gold_df["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
gold_df = gold_df.rename(columns={"dropout_pred": "dropout_predicted"})

GOLD_COLUMNS = [
    "student_id", "risk_score", "dropout_predicted", "intervention_tier",
    "shap_factor_1", "shap_value_1", "shap_factor_2", "shap_value_2",
    "shap_factor_3", "shap_value_3", "reason_text",
    "gender", "financial_stress_index", "grade_delta", "scored_at",
]

gold_output = gold_df[GOLD_COLUMNS].copy()
gold_output["student_id"] = gold_output["student_id"].astype(int)
gold_output["dropout_predicted"] = gold_output["dropout_predicted"].astype(int)

print(f"✅ Gold schema: {gold_output.shape[1]} columns, {gold_output.shape[0]} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.6 Validation

# COMMAND ----------

all_clean = True
for col in GOLD_COLUMNS:
    n = gold_output[col].isnull().sum()
    if n > 0: all_clean = False
    print(f"  {'✅' if n==0 else '❌'} {col}: {n} nulls")

tiers = set(gold_output["intervention_tier"].unique())
print(f"\nTiers: {tiers} {'✅' if tiers <= {'high','medium','low'} else '❌'}")
print(f"Risk range: [{gold_output['risk_score'].min():.4f}, {gold_output['risk_score'].max():.4f}]")
print(f"Short reason_text: {(gold_output['reason_text'].str.len() < 10).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.7 Write Gold Delta Table

# COMMAND ----------

gold_spark = spark.createDataFrame(gold_output)
gold_spark.write.format("delta").mode("overwrite").saveAsTable("gold.at_risk_students")

verify = spark.table("gold.at_risk_students")
print(f"✅ gold.at_risk_students written: {verify.count()} rows, {len(verify.columns)} columns")

# COMMAND ----------

display(spark.table("gold.at_risk_students").orderBy("risk_score", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.8 Summary

# COMMAND ----------

total = len(gold_output)
high = len(gold_output[gold_output["intervention_tier"] == "high"])
medium = len(gold_output[gold_output["intervention_tier"] == "medium"])
low = len(gold_output[gold_output["intervention_tier"] == "low"])

print("=" * 70)
print("GOLD TABLE SUMMARY")
print("=" * 70)
print(f"  Total at-risk: {total}")
print(f"  🔴 HIGH:   {high} ({high/total*100:.1f}%)")
print(f"  🟡 MEDIUM: {medium} ({medium/total*100:.1f}%)")
print(f"  🟢 LOW:    {low} ({low/total*100:.1f}%)")
print(f"  Avg risk:  {gold_output['risk_score'].mean():.3f}")
print(f"\n  ★ All scores Platt-calibrated")
print(f"  ★ All students have reason_text")
print(f"  ★ All students have SHAP top-3")

# COMMAND ----------

# Sample rows for presentation
print("\nSAMPLE ROWS FOR PRESENTATION:")
for _, r in gold_output.sort_values("risk_score", ascending=False).head(5).iterrows():
    print(f"\n  Student {int(r['student_id'])} | {r['risk_score']:.3f} | 🔴 {r['intervention_tier'].upper()}")
    print(f"  SHAP: {r['shap_factor_1']} ({r['shap_value_1']:+.3f}), {r['shap_factor_2']} ({r['shap_value_2']:+.3f}), {r['shap_factor_3']} ({r['shap_value_3']:+.3f})")
    print(f"  ★ {r['reason_text']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Pipeline Complete!
# MAGIC
# MAGIC | Table | Status |
# MAGIC |-------|--------|
# MAGIC | `bronze.uci_dropout` | ✅ |
# MAGIC | `silver.uci_dropout_clean` | ✅ |
# MAGIC | `audit.fairness_metrics` | ✅ |
# MAGIC | `gold.at_risk_students` | ✅ |
# MAGIC
# MAGIC | MLflow | Status |
# MAGIC |--------|--------|
# MAGIC | `logistic_regression_baseline` | ✅ |
# MAGIC | `xgboost_classifier` | ✅ |
# MAGIC | `dropout_risk_champion` (Production) | ✅ Calibrated |
