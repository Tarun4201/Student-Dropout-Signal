# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 02 — Silver Layer
# MAGIC ## The Dropout Signal: Data Cleaning & Feature Engineering
# MAGIC
# MAGIC **Input Table:** `bronze.uci_dropout`
# MAGIC
# MAGIC **Output Table:** `silver.uci_dropout_clean`
# MAGIC
# MAGIC **Engineered Features:** `grade_delta`, `absenteeism_trend`, `financial_stress_index`, `engagement_score`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Read Bronze Table

# COMMAND ----------

df = spark.table("bronze.uci_dropout")
print(f"✅ Bronze table loaded: {df.count()} rows, {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Verify Column Names

# COMMAND ----------

print("Current column names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Fix Data Type Issues
# MAGIC
# MAGIC Cast `curricular_units_2nd_sem_approved` from string to double (contains malformed value 'Dropout' in source data).

# COMMAND ----------

from pyspark.sql import functions as F

df = df.withColumn(
    "curricular_units_2nd_sem_approved",
    F.expr("try_cast(curricular_units_2nd_sem_approved as double)")
)

print("✅ curricular_units_2nd_sem_approved cast to double")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Null Handling
# MAGIC
# MAGIC - Numeric: median imputation
# MAGIC - String: mode imputation
# MAGIC - No rows dropped

# COMMAND ----------

from pyspark.sql.types import StringType

# Check nulls before
print("Null counts BEFORE imputation:")
print("=" * 50)
total_nulls = 0
for col_name in df.columns:
    null_count = df.filter(F.col(col_name).isNull()).count()
    total_nulls += null_count
    if null_count > 0:
        print(f"  {col_name:<55}: {null_count} nulls")

if total_nulls == 0:
    print("  No nulls found in any column! ✅")
else:
    print(f"\n  Total nulls: {total_nulls}")

# COMMAND ----------

# Impute nulls (defensive coding even if none found)
for col_name in df.columns:
    col_type = df.schema[col_name].dataType
    if isinstance(col_type, StringType):
        mode_val = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
        if mode_val:
            df = df.fillna({col_name: mode_val[0]})
    else:
        median_val = df.approxQuantile(col_name, [0.5], 0.001)
        if median_val:
            df = df.fillna({col_name: median_val[0]})

# Verify
total_nulls_after = sum(df.filter(F.col(c).isNull()).count() for c in df.columns)
print(f"✅ Null counts AFTER imputation: {total_nulls_after} (should be 0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Binary Target Encoding
# MAGIC
# MAGIC `dropout_label`: **1** if `target == "Dropout"`, else **0**.

# COMMAND ----------

df = df.withColumn(
    "dropout_label",
    F.when(F.col("target") == "Dropout", 1).otherwise(0)
)

print("Binary target distribution:")
df.groupBy("dropout_label").count().orderBy("dropout_label").show()

total = df.count()
pos = df.filter(F.col("dropout_label") == 1).count()
neg = total - pos
print(f"  Dropout (1): {pos} ({pos/total*100:.1f}%)")
print(f"  Non-dropout (0): {neg} ({neg/total*100:.1f}%)")
print(f"  scale_pos_weight for XGBoost: {neg/pos:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 Feature Engineering
# MAGIC
# MAGIC ### Feature 1: `grade_delta`
# MAGIC `sem2_grade − sem1_grade` — Negative = declining performance

# COMMAND ----------

df = df.withColumn(
    "grade_delta",
    F.col("curricular_units_2nd_sem_grade") - F.col("curricular_units_1st_sem_grade")
)

print("✅ grade_delta created")
df.select("curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade", "grade_delta").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 2: `absenteeism_trend`
# MAGIC `(enr1 − app1 + enr2 − app2) / (enr1 + enr2 + 1)` — Higher = more disengagement

# COMMAND ----------

df = df.withColumn(
    "absenteeism_trend",
    (F.col("curricular_units_1st_sem_enrolled") - F.col("curricular_units_1st_sem_approved") +
     F.col("curricular_units_2nd_sem_enrolled") - F.col("curricular_units_2nd_sem_approved")) /
    (F.col("curricular_units_1st_sem_enrolled") + F.col("curricular_units_2nd_sem_enrolled") + 1)
)

print("✅ absenteeism_trend created")
df.select("curricular_units_1st_sem_enrolled", "curricular_units_1st_sem_approved",
          "curricular_units_2nd_sem_enrolled", "curricular_units_2nd_sem_approved",
          "absenteeism_trend").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 3: `financial_stress_index`
# MAGIC `debtor×2 + (1−tuition_fees_up_to_date)×2 + (1−scholarship_holder)` — Range 0–5

# COMMAND ----------

df = df.withColumn(
    "financial_stress_index",
    F.col("debtor") * 2 +
    (1 - F.col("tuition_fees_up_to_date")) * 2 +
    (1 - F.col("scholarship_holder"))
)

print("✅ financial_stress_index created (range 0–5)")
df.groupBy("financial_stress_index").count().orderBy("financial_stress_index").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature 4: `engagement_score`
# MAGIC `(app1/(enr1+1)) + (app2/(enr2+1)) + (eval1+eval2)/20`

# COMMAND ----------

df = df.withColumn(
    "engagement_score",
    (F.col("curricular_units_1st_sem_approved") / (F.col("curricular_units_1st_sem_enrolled") + 1)) +
    (F.col("curricular_units_2nd_sem_approved") / (F.col("curricular_units_2nd_sem_enrolled") + 1)) +
    (F.col("curricular_units_1st_sem_evaluations") + F.col("curricular_units_2nd_sem_evaluations")) / 20
)

print("✅ engagement_score created")
df.select("engagement_score").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7 Add Synthetic Student ID

# COMMAND ----------

from pyspark.sql.window import Window

df = df.withColumn(
    "student_id",
    F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1
)

print(f"✅ student_id added (range: 0 to {df.count() - 1})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.8 Final Validation

# COMMAND ----------

print("Final null check:")
any_nulls = False
for col_name in df.columns:
    null_count = df.filter(F.col(col_name).isNull()).count()
    if null_count > 0:
        print(f"  ⚠️  {col_name}: {null_count} nulls")
        any_nulls = True
if not any_nulls:
    print("  ✅ Zero nulls in all columns")

required = ["dropout_label", "grade_delta", "absenteeism_trend",
            "financial_stress_index", "engagement_score", "student_id", "target"]
print(f"\nRequired columns check:")
for col in required:
    exists = col in df.columns
    print(f"  {'✅' if exists else '❌'} {col}")

distinct_labels = [row.dropout_label for row in df.select("dropout_label").distinct().collect()]
print(f"\ndropout_label distinct values: {sorted(distinct_labels)} (should be [0, 1])")
print(f"\nSilver table shape: {df.count()} rows × {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9 Write Silver Delta Table

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("silver.uci_dropout_clean")

verify_df = spark.table("silver.uci_dropout_clean")
print("✅ Silver table written: silver.uci_dropout_clean")
print(f"   Rows: {verify_df.count()}")
print(f"   Columns: {len(verify_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.10 Silver Table Preview

# COMMAND ----------

display(spark.table("silver.uci_dropout_clean").select(
    "student_id", "target", "dropout_label", "gender", "age_at_enrollment",
    "grade_delta", "absenteeism_trend", "financial_stress_index", "engagement_score"
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Silver Layer Complete
# MAGIC
# MAGIC **Next:** Run `03_model_training` to train models and register the champion.