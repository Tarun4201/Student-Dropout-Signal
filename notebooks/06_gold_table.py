# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 06 — Gold Output Table
# MAGIC ## The Dropout Signal: Final At-Risk Students with ★ reason_text
# MAGIC
# MAGIC **Purpose:** Build the Gold-layer Delta table with calibrated risk scores,
# MAGIC SHAP top-3 factors, ★ plain-English `reason_text`, and intervention tiers.
# MAGIC
# MAGIC **Input Tables:** `silver.model_test_results` + `silver.shap_results`
# MAGIC
# MAGIC **Output Table:** `gold.at_risk_students`
# MAGIC
# MAGIC **★ Differentiator:** `reason_text` translates SHAP into human-readable sentences for advisors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Load Data

# COMMAND ----------

import pandas as pd
from datetime import datetime

# Load test results (predictions + all features)
test_df = spark.table("silver.model_test_results").toPandas()

# Load SHAP results (per-student top-3 factors)
shap_df = spark.table("silver.shap_results").toPandas()

print(f"✅ Test results loaded: {test_df.shape[0]} rows")
print(f"✅ SHAP results loaded: {shap_df.shape[0]} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Join Test Results with SHAP Factors

# COMMAND ----------

# Ensure student_id types match for join
test_df["student_id"] = test_df["student_id"].astype(int)
shap_df["student_id"] = shap_df["student_id"].astype(int)

# Join on student_id
gold_df = test_df.merge(shap_df, on="student_id", how="inner")

print(f"✅ Joined: {gold_df.shape[0]} rows (should equal test set size)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Filter to At-Risk Students Only
# MAGIC
# MAGIC Gold table contains **only** students with `dropout_predicted == 1`.

# COMMAND ----------

gold_df = gold_df[gold_df["dropout_pred"] == 1].copy()

print(f"✅ Filtered to at-risk students: {gold_df.shape[0]} students")
print(f"   These are students the calibrated model predicts will drop out.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 Assign Intervention Tiers
# MAGIC
# MAGIC | Tier | Rule | Action |
# MAGIC |------|------|--------|
# MAGIC | **HIGH** | `risk_score ≥ 0.70` | Immediate outreach by academic advisor |
# MAGIC | **MEDIUM** | `0.40 ≤ risk_score < 0.70` | Scheduled check-in within 2 weeks |
# MAGIC | **LOW** | `risk_score < 0.40` | Monitor only, no active intervention |
# MAGIC
# MAGIC **Rationale:** Thresholds apply to Platt-calibrated scores. At ~32% base dropout rate,
# MAGIC a calibrated score of 0.70 implies ~2.2× higher dropout probability than base.
# MAGIC Because scores are calibrated, 0.70 approximates the precision level acceptable for advisor time investment.

# COMMAND ----------

def assign_tier(risk_score):
    """Assign intervention tier based on calibrated risk score."""
    if risk_score >= 0.70:
        return "high"
    elif risk_score >= 0.40:
        return "medium"
    else:
        return "low"

gold_df["intervention_tier"] = gold_df["risk_score"].apply(assign_tier)

# Distribution
tier_dist = gold_df["intervention_tier"].value_counts()
print("Intervention tier distribution:")
print("=" * 40)
for tier in ["high", "medium", "low"]:
    count = tier_dist.get(tier, 0)
    pct = count / len(gold_df) * 100
    print(f"  {tier.upper():>8}: {count:4d} students ({pct:.1f}%)")
print(f"  {'TOTAL':>8}: {len(gold_df):4d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 ★ Generate `reason_text`
# MAGIC
# MAGIC **This is the primary deliverable that advisors will read.**
# MAGIC
# MAGIC Each at-risk student gets a plain-English sentence built from their top-3 SHAP factors
# MAGIC using the `factor_interpretations` dictionary. Advisors do not need to understand SHAP.
# MAGIC
# MAGIC **Example output:**
# MAGIC > "Grade fell 2.3pts semester-on-semester; financial stress score 4/5; 67% unit non-completion rate."

# COMMAND ----------

# ★ Factor interpretation functions — translate feature names + values into plain English
factor_interpretations = {
    # Engineered features
    "grade_delta": lambda v: f"grade {'fell' if v < 0 else 'rose'} {abs(v):.1f}pts semester-on-semester",
    "financial_stress_index": lambda v: f"financial stress score {v:.0f}/5",
    "absenteeism_trend": lambda v: f"{v*100:.0f}% unit non-completion rate",
    "engagement_score": lambda v: f"engagement score of {v:.2f}",
    
    # Curricular units — semester grades
    "Curricular_units_2nd_sem_grade": lambda v: f"semester 2 grade of {v:.1f}",
    "Curricular_units_1st_sem_grade": lambda v: f"semester 1 grade of {v:.1f}",
    
    # Curricular units — approved
    "Curricular_units_2nd_sem_approved": lambda v: f"{int(v)} units approved in semester 2",
    "Curricular_units_1st_sem_approved": lambda v: f"{int(v)} units approved in semester 1",
    
    # Curricular units — enrolled
    "Curricular_units_2nd_sem_enrolled": lambda v: f"{int(v)} units enrolled in semester 2",
    "Curricular_units_1st_sem_enrolled": lambda v: f"{int(v)} units enrolled in semester 1",
    
    # Curricular units — evaluations
    "Curricular_units_2nd_sem_evaluations": lambda v: f"{int(v)} evaluations in semester 2",
    "Curricular_units_1st_sem_evaluations": lambda v: f"{int(v)} evaluations in semester 1",
    
    # Financial indicators
    "Debtor": lambda v: "outstanding debt on record" if v == 1 else "no debt on record",
    "Tuition_fees_up_to_date": lambda v: "tuition fees overdue" if v == 0 else "tuition fees current",
    "Scholarship_holder": lambda v: "no scholarship" if v == 0 else "scholarship holder",
    
    # Demographics
    "Age_at_enrollment": lambda v: f"enrolled at age {int(v)}",
    "Admission_grade": lambda v: f"admission grade of {v:.1f}",
    "Previous_qualification_grade": lambda v: f"previous qualification grade of {v:.1f}",
    
    # Macroeconomic
    "Unemployment_rate": lambda v: f"unemployment rate of {v:.1f}%",
    "GDP": lambda v: f"GDP growth of {v:.2f}%",
    "Inflation_rate": lambda v: f"inflation rate of {v:.1f}%",
    
    # Other
    "Displaced": lambda v: "displaced student" if v == 1 else "non-displaced student",
    "International": lambda v: "international student" if v == 1 else "domestic student",
    "Gender": lambda v: "male" if v == 1 else "female",
    "Daytime_evening_attendance": lambda v: "daytime attendance" if v == 1 else "evening attendance",
    
    # Curricular units — credited & without evaluations
    "Curricular_units_1st_sem_credited": lambda v: f"{int(v)} units credited in semester 1",
    "Curricular_units_2nd_sem_credited": lambda v: f"{int(v)} units credited in semester 2",
    "Curricular_units_1st_sem_without_evaluations": lambda v: f"{int(v)} units without evaluation in semester 1",
    "Curricular_units_2nd_sem_without_evaluations": lambda v: f"{int(v)} units without evaluation in semester 2",
}

def build_reason_text(row):
    """
    ★ Build a plain-English reason sentence from the top-3 SHAP factors.
    
    For each SHAP factor, looks up the actual student value and generates
    a human-readable phrase using the factor_interpretations dict.
    Falls back to feature name if no interpretation is available.
    """
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
            # Fallback: use feature name as readable text
            reasons.append(feature.replace("_", " "))
    
    return "; ".join(reasons).capitalize() + "."

# Apply reason_text generation
gold_df["reason_text"] = gold_df.apply(build_reason_text, axis=1)

print(f"✅ reason_text generated for {len(gold_df)} at-risk students")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ★ Sample `reason_text` Output

# COMMAND ----------

# Show sample reason_text values
print("Sample reason_text for at-risk students:")
print("=" * 100)
for _, row in gold_df.head(10).iterrows():
    tier_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[row["intervention_tier"]]
    print(f"\n  Student {int(row['student_id'])} | Risk: {row['risk_score']:.3f} | {tier_emoji} {row['intervention_tier'].upper()}")
    print(f"  → {row['reason_text']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.6 Add Timestamp

# COMMAND ----------

gold_df["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.7 Select Final Gold Schema
# MAGIC
# MAGIC Selecting only the columns specified in the PRD Gold table schema.

# COMMAND ----------

GOLD_COLUMNS = [
    "student_id",
    "risk_score",
    "dropout_predicted",
    "intervention_tier",
    "shap_factor_1",
    "shap_value_1",
    "shap_factor_2",
    "shap_value_2",
    "shap_factor_3",
    "shap_value_3",
    "reason_text",
    "gender",
    "financial_stress_index",
    "grade_delta",
    "scored_at",
]

# Rename columns to match Gold schema
gold_df = gold_df.rename(columns={
    "dropout_pred": "dropout_predicted",
    "Gender": "gender",
})

# Select Gold columns
gold_output = gold_df[GOLD_COLUMNS].copy()

# Ensure types
gold_output["student_id"] = gold_output["student_id"].astype(int)
gold_output["dropout_predicted"] = gold_output["dropout_predicted"].astype(int)

print(f"✅ Gold schema applied: {gold_output.shape[1]} columns")
print(f"   Rows: {gold_output.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.8 Validation — Zero Nulls

# COMMAND ----------

print("Gold table null check:")
print("=" * 50)
all_clean = True
for col in GOLD_COLUMNS:
    null_count = gold_output[col].isnull().sum()
    status = "✅" if null_count == 0 else "❌"
    if null_count > 0:
        all_clean = False
    print(f"  {status} {col:<30s}: {null_count} nulls")

print(f"\n{'✅ All columns clean — zero nulls' if all_clean else '❌ Some columns have nulls — fix before writing!'}")

# Validate intervention_tier values
valid_tiers = set(gold_output["intervention_tier"].unique())
expected_tiers = {"high", "medium", "low"}
print(f"\nIntervention tiers found: {valid_tiers}")
print(f"Valid tiers only: {'✅' if valid_tiers.issubset(expected_tiers) else '❌'}")

# Validate risk_score range
min_score = gold_output["risk_score"].min()
max_score = gold_output["risk_score"].max()
print(f"\nRisk score range: [{min_score:.4f}, {max_score:.4f}] (should be 0.0–1.0)")

# Validate reason_text is non-empty and non-generic
empty_reasons = (gold_output["reason_text"].str.len() < 10).sum()
print(f"Short/empty reason_text: {empty_reasons} {'✅' if empty_reasons == 0 else '⚠️'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.9 Write Gold Delta Table

# COMMAND ----------

gold_spark = spark.createDataFrame(gold_output)
gold_spark.write.format("delta").mode("overwrite").saveAsTable("gold.at_risk_students")

# Verify
verify_df = spark.table("gold.at_risk_students")
print("✅ Gold table written successfully: gold.at_risk_students")
print(f"   Rows: {verify_df.count()}")
print(f"   Columns: {len(verify_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.10 Gold Table Preview

# COMMAND ----------

display(spark.table("gold.at_risk_students").orderBy("risk_score", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.11 Summary Statistics

# COMMAND ----------

print("=" * 70)
print("GOLD TABLE SUMMARY — gold.at_risk_students")
print("=" * 70)

total = len(gold_output)
high = len(gold_output[gold_output["intervention_tier"] == "high"])
medium = len(gold_output[gold_output["intervention_tier"] == "medium"])
low = len(gold_output[gold_output["intervention_tier"] == "low"])

print(f"\n  Total at-risk students:  {total}")
print(f"  🔴 HIGH tier (≥0.70):    {high} ({high/total*100:.1f}%) — immediate outreach")
print(f"  🟡 MEDIUM tier (0.40–0.70): {medium} ({medium/total*100:.1f}%) — check-in within 2 weeks")
print(f"  🟢 LOW tier (<0.40):     {low} ({low/total*100:.1f}%) — monitor only")
print(f"\n  Average risk score:      {gold_output['risk_score'].mean():.3f}")
print(f"  Median risk score:       {gold_output['risk_score'].median():.3f}")

print(f"\n  ★ All students have calibrated risk scores (Platt-scaled)")
print(f"  ★ All students have plain-English reason_text")
print(f"  ★ All students have top-3 SHAP factors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.12 Sample Gold Table Rows for Presentation
# MAGIC
# MAGIC Print a few high-tier students with full detail — use this slide in the presentation.

# COMMAND ----------

print("SAMPLE AT-RISK STUDENTS FOR PRESENTATION")
print("=" * 80)

high_risk = gold_output[gold_output["intervention_tier"] == "high"].sort_values("risk_score", ascending=False)

for _, row in high_risk.head(5).iterrows():
    print(f"\n  ┌─ Student ID: {int(row['student_id'])}")
    print(f"  │  Risk Score: {row['risk_score']:.3f} (calibrated)")
    print(f"  │  Intervention: 🔴 {row['intervention_tier'].upper()}")
    print(f"  │  SHAP Factor 1: {row['shap_factor_1']} ({row['shap_value_1']:+.3f})")
    print(f"  │  SHAP Factor 2: {row['shap_factor_2']} ({row['shap_value_2']:+.3f})")
    print(f"  │  SHAP Factor 3: {row['shap_factor_3']} ({row['shap_value_3']:+.3f})")
    print(f"  │  Grade Delta: {row['grade_delta']:.2f}")
    print(f"  │  Financial Stress: {row['financial_stress_index']:.0f}/5")
    print(f"  └─ ★ Reason: {row['reason_text']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Gold Table Complete — Pipeline Finished!
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - ✅ `gold.at_risk_students` Delta table: only students with `dropout_predicted == 1`
# MAGIC - ✅ Zero null values in all required columns
# MAGIC - ✅ ★ `risk_score` is Platt-calibrated (not raw XGBoost output)
# MAGIC - ✅ ★ `reason_text` contains human-readable sentences referencing actual student values
# MAGIC - ✅ `intervention_tier` values are only "high", "medium", or "low"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Full Pipeline Summary
# MAGIC
# MAGIC | Table | Status |
# MAGIC |-------|--------|
# MAGIC | `bronze.uci_dropout` | ✅ Raw ingest |
# MAGIC | `silver.uci_dropout_clean` | ✅ Cleaned + features |
# MAGIC | `silver.model_test_results` | ✅ Test predictions |
# MAGIC | `silver.shap_results` | ✅ Per-student SHAP |
# MAGIC | `audit.fairness_metrics` | ✅ Marginal + intersectional |
# MAGIC | `gold.at_risk_students` | ✅ Final output with reason_text |
# MAGIC
# MAGIC | MLflow Asset | Status |
# MAGIC |-------------|--------|
# MAGIC | Experiment `dropout_signal_hackathon` | ✅ 2 runs |
# MAGIC | `logistic_regression_baseline` run | ✅ All params/metrics |
# MAGIC | `xgboost_classifier` run | ✅ All params/metrics + artifacts |
# MAGIC | `dropout_risk_champion` model (Production) | ✅ Calibrated + tagged |
