# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 02 — Silver Layer
# MAGIC ## The Dropout Signal: Data Cleaning & Feature Engineering
# MAGIC
# MAGIC **Purpose:** Clean the Bronze data, rename columns, encode the binary target, and engineer all required features.
# MAGIC
# MAGIC **Input Table:** `bronze.uci_dropout`
# MAGIC
# MAGIC **Output Table:** `silver.uci_dropout_clean`
# MAGIC
# MAGIC **Engineered Features:**
# MAGIC - `grade_delta` — semester grade trend
# MAGIC - `absenteeism_trend` — unit non-completion rate
# MAGIC - `financial_stress_index` — composite financial risk (0–5)
# MAGIC - `engagement_score` — composite engagement metric

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Read Bronze Table

# COMMAND ----------

df = spark.table("bronze.uci_dropout")
print(f"✅ Bronze table loaded: {df.count()} rows, {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Column Renaming
# MAGIC
# MAGIC The original CSV has column names with spaces, parentheses, and apostrophes.
# MAGIC We rename them to clean underscore format for easier downstream processing.
# MAGIC
# MAGIC **No data values are modified — only column headers are renamed.**

# COMMAND ----------

import re

def clean_col_name(name):
    """
    Clean column name: remove apostrophes, replace / and spaces with underscores,
    remove parentheses. Capitalize 'target' to 'Target' per PRD convention.
    """
    name = name.replace("'", "")         # Mother's → Mothers
    name = name.replace("/", "_")        # Daytime/evening → Daytime_evening
    name = name.replace("(", "")         # Remove (
    name = name.replace(")", "")         # Remove )
    name = name.strip()                   # Trim whitespace
    name = re.sub(r'\s+', '_', name)     # Spaces → underscores
    if name.lower() == 'target':
        name = 'Target'                   # Capitalize to match PRD
    return name

# Apply renaming
print("Column renaming:")
print(f"{'Original':<55} → {'Cleaned'}")
print("=" * 100)

for old_name in df.columns:
    new_name = clean_col_name(old_name)
    if old_name != new_name:
        df = df.withColumnRenamed(old_name, new_name)
        print(f"{old_name:<55} → {new_name}")

print(f"\n✅ All {len(df.columns)} columns renamed successfully")

# COMMAND ----------

# Verify cleaned column names
print("Cleaned column names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Null Handling
# MAGIC
# MAGIC **Strategy (per PRD):**
# MAGIC - Numeric columns: impute with column **median** (robust to outliers)
# MAGIC - Categorical/string columns: impute with mode or "Unknown"
# MAGIC - **Do not drop rows** — 4,400 records is small; every row is valuable

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, LongType

# Document null counts BEFORE imputation
print("Null counts BEFORE imputation:")
print("=" * 50)
null_counts_before = {}
for col_name in df.columns:
    null_count = df.filter(F.col(col_name).isNull()).count()
    null_counts_before[col_name] = null_count
    if null_count > 0:
        print(f"  {col_name:<55}: {null_count} nulls")

total_nulls = sum(null_counts_before.values())
if total_nulls == 0:
    print("  No nulls found in any column! ✅")
else:
    print(f"\n  Total nulls across all columns: {total_nulls}")

# COMMAND ----------

# Impute nulls (even if none found — defensive coding)
from pyspark.sql.types import NumericType

for col_name in df.columns:
    col_type = df.schema[col_name].dataType
    
    if isinstance(col_type, StringType):
        # String columns: fill with mode
        mode_val = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
        if mode_val:
            df = df.fillna({col_name: mode_val[0]})
    else:
        # Numeric columns: fill with median
        median_val = df.approxQuantile(col_name, [0.5], 0.001)
        if median_val:
            df = df.fillna({col_name: median_val[0]})

# Verify: null counts AFTER imputation
total_nulls_after = 0
for col_name in df.columns:
    null_count = df.filter(F.col(col_name).isNull()).count()
    total_nulls_after += null_count

print(f"✅ Null counts AFTER imputation: {total_nulls_after} (should be 0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Binary Target Encoding
# MAGIC
# MAGIC Create `dropout_label`: **1** where `Target == "Dropout"`, else **0**.
# MAGIC
# MAGIC The original `Target` column is retained per PRD requirement.

# COMMAND ----------

df = df.withColumn(
    "dropout_label",
    F.when(F.col("Target") == "Dropout", 1).otherwise(0)
)

# Verify class balance
print("Binary target distribution:")
print("=" * 40)
df.groupBy("dropout_label").count().orderBy("dropout_label").show()

total = df.count()
pos = df.filter(F.col("dropout_label") == 1).count()
neg = total - pos
print(f"  Dropout (1): {pos} ({pos/total*100:.1f}%)")
print(f"  Non-dropout (0): {neg} ({neg/total*100:.1f}%)")
print(f"  Imbalance ratio: 1:{neg/pos:.1f}")
print(f"  scale_pos_weight for XGBoost: {neg/pos:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Feature Engineering
# MAGIC
# MAGIC ### Feature 1: `grade_delta`
# MAGIC **Formula:** `sem2_grade − sem1_grade`
# MAGIC
# MAGIC **Interpretation:** Negative = declining academic performance (strong dropout signal)

# COMMAND ----------

df = df.withColumn(
    "grade_delta",
    F.col("Curricular_units_2nd_sem_grade") - F.col("Curricular_units_1st_sem_grade")
)

print("✅ grade_delta created")
print("   Sample values:")
df.select("Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade", "grade_delta").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 2: `absenteeism_trend`
# MAGIC **Formula:** `(enr1 − app1 + enr2 − app2) / (enr1 + enr2 + 1)`
# MAGIC
# MAGIC **Interpretation:** Rate of enrolling but not completing units. Higher = more disengagement.

# COMMAND ----------

df = df.withColumn(
    "absenteeism_trend",
    (F.col("Curricular_units_1st_sem_enrolled") - F.col("Curricular_units_1st_sem_approved") +
     F.col("Curricular_units_2nd_sem_enrolled") - F.col("Curricular_units_2nd_sem_approved")) /
    (F.col("Curricular_units_1st_sem_enrolled") + F.col("Curricular_units_2nd_sem_enrolled") + 1)
)

print("✅ absenteeism_trend created")
print("   Sample values:")
df.select("Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved",
          "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_approved",
          "absenteeism_trend").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 3: `financial_stress_index`
# MAGIC **Formula:** `Debtor × 2 + (1 − Tuition_fees_up_to_date) × 2 + (1 − Scholarship_holder)`
# MAGIC
# MAGIC **Range:** 0–5 (higher = more financial stress)
# MAGIC
# MAGIC **Weighting rationale:** Debt and overdue fees carry 2× weight because they have stronger
# MAGIC empirical correlation with dropout than absence of scholarship alone.

# COMMAND ----------

df = df.withColumn(
    "financial_stress_index",
    F.col("Debtor") * 2 +
    (1 - F.col("Tuition_fees_up_to_date")) * 2 +
    (1 - F.col("Scholarship_holder"))
)

print("✅ financial_stress_index created")
print("   Range: 0–5")
print("   Distribution:")
df.groupBy("financial_stress_index").count().orderBy("financial_stress_index").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 4 (Bonus): `engagement_score`
# MAGIC **Formula:** `(app1 / (enr1 + 1)) + (app2 / (enr2 + 1)) + (eval1 + eval2) / 20`
# MAGIC
# MAGIC **Interpretation:** Composite of unit completion rate + assessment participation.
# MAGIC Higher = more engaged student.

# COMMAND ----------

df = df.withColumn(
    "engagement_score",
    (F.col("Curricular_units_1st_sem_approved") / (F.col("Curricular_units_1st_sem_enrolled") + 1)) +
    (F.col("Curricular_units_2nd_sem_approved") / (F.col("Curricular_units_2nd_sem_enrolled") + 1)) +
    (F.col("Curricular_units_1st_sem_evaluations") + F.col("Curricular_units_2nd_sem_evaluations")) / 20
)

print("✅ engagement_score created")
print("   Sample values:")
df.select("engagement_score").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 Add Synthetic Student ID
# MAGIC
# MAGIC Add `student_id` as a row index for tracking individual students through the pipeline.

# COMMAND ----------

from pyspark.sql.window import Window

df = df.withColumn(
    "student_id",
    F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1
)

print(f"✅ student_id added (range: 0 to {df.count() - 1})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7 Final Validation

# COMMAND ----------

# Verify no nulls in any column
print("Final null check:")
print("=" * 50)
any_nulls = False
for col_name in df.columns:
    null_count = df.filter(F.col(col_name).isNull()).count()
    if null_count > 0:
        print(f"  ⚠️  {col_name}: {null_count} nulls")
        any_nulls = True

if not any_nulls:
    print("  ✅ Zero nulls in all columns")

# Verify required columns exist
required = ["dropout_label", "grade_delta", "absenteeism_trend", 
            "financial_stress_index", "engagement_score", "student_id", "Target"]
print(f"\nRequired columns check:")
for col in required:
    exists = col in df.columns
    print(f"  {'✅' if exists else '❌'} {col}")

# Verify dropout_label values
distinct_labels = [row.dropout_label for row in df.select("dropout_label").distinct().collect()]
print(f"\ndropout_label distinct values: {sorted(distinct_labels)} (should be [0, 1])")

# Final shape
print(f"\nSilver table shape: {df.count()} rows × {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.8 Write Silver Delta Table

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("silver.uci_dropout_clean")

# Verify
verify_df = spark.table("silver.uci_dropout_clean")
print("✅ Silver table written successfully: silver.uci_dropout_clean")
print(f"   Rows: {verify_df.count()}")
print(f"   Columns: {len(verify_df.columns)}")
print(f"   New columns: dropout_label, grade_delta, absenteeism_trend, financial_stress_index, engagement_score, student_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9 Silver Table Preview

# COMMAND ----------

display(spark.table("silver.uci_dropout_clean").select(
    "student_id", "Target", "dropout_label", "Gender", "Age_at_enrollment",
    "grade_delta", "absenteeism_trend", "financial_stress_index", "engagement_score"
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Silver Layer Complete
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - `silver.uci_dropout_clean` Delta table with cleaned column names
# MAGIC - `dropout_label` binary target (1 = Dropout, 0 = Not)
# MAGIC - All 4 engineered features present with documented formulas
# MAGIC - Zero nulls across all columns
# MAGIC - Original `Target` column retained
# MAGIC
# MAGIC **Next:** Run `03_model_training` to train models and register the champion.
