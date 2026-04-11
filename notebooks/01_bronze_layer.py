# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 01 — Bronze Layer
# MAGIC ## The Dropout Signal: Raw Data Ingest
# MAGIC
# MAGIC **Purpose:** Copy the uploaded dataset into a governed Bronze Delta table with a documented schema.
# MAGIC
# MAGIC **Source Table:** `workspace.default.students_dropout_academic_success`
# MAGIC
# MAGIC **Output Table:** `bronze.uci_dropout`
# MAGIC
# MAGIC **Rules:** Zero transformations — Bronze is always raw and immutable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Create Schemas

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
spark.sql("CREATE SCHEMA IF NOT EXISTS audit")

print("✅ All schemas created: bronze, silver, gold, audit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Read Source Table
# MAGIC
# MAGIC The dataset was uploaded via Databricks UI and is available as a Delta table
# MAGIC in the catalog at `workspace.default.students_dropout_academic_success`.

# COMMAND ----------

# Read from the existing catalog table
df_bronze = spark.table("workspace.default.students_dropout_academic_success")

row_count = df_bronze.count()
col_count = len(df_bronze.columns)
print(f"✅ Source table loaded successfully")
print(f"   Rows: {row_count}")
print(f"   Columns: {col_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Schema — `printSchema()`

# COMMAND ----------

df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Preview — First 10 Rows

# COMMAND ----------

display(df_bronze.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Column Documentation
# MAGIC
# MAGIC Automated documentation: column name, data type, null count for every column.

# COMMAND ----------

from pyspark.sql import functions as F

print(f"{'#':<4} {'Column Name':<55} {'Type':<14} {'Nulls'}")
print("=" * 90)

for idx, col_name in enumerate(df_bronze.columns, 1):
    col_type = str(df_bronze.schema[col_name].dataType).replace("Type", "")
    null_count = df_bronze.filter(F.col(col_name).isNull()).count()
    print(f"{idx:<4} {col_name:<55} {col_type:<14} {null_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 Column Reference (Plain-English Meanings)
# MAGIC
# MAGIC | # | Column | Meaning |
# MAGIC |---|--------|---------|
# MAGIC | 1 | marital_status | Marital status code (1=single, 2=married, etc.) |
# MAGIC | 2 | application_mode | Method of application (1=general, 17=post-secondary, etc.) |
# MAGIC | 3 | application_order | Preference order of this course (1=first choice through 9) |
# MAGIC | 4 | course | Course code identifier |
# MAGIC | 5 | daytime_evening_attendance | 1=daytime, 0=evening |
# MAGIC | 6 | previous_qualification | Code for previous qualification type |
# MAGIC | 7 | previous_qualification_grade | Grade from previous qualification (0-200) |
# MAGIC | 8 | nationality | Nationality code (1=Portuguese) |
# MAGIC | 9 | mothers_qualification | Mother's education level code |
# MAGIC | 10 | fathers_qualification | Father's education level code |
# MAGIC | 11 | mothers_occupation | Mother's occupation code |
# MAGIC | 12 | fathers_occupation | Father's occupation code |
# MAGIC | 13 | admission_grade | Admission exam grade (0-200) |
# MAGIC | 14 | displaced | 1=displaced from home region, 0=not |
# MAGIC | 15 | educational_special_needs | 1=has special needs, 0=not |
# MAGIC | 16 | debtor | 1=has outstanding debt, 0=not |
# MAGIC | 17 | tuition_fees_up_to_date | 1=fees current, 0=fees overdue |
# MAGIC | 18 | gender | 1=male, 0=female |
# MAGIC | 19 | scholarship_holder | 1=receives scholarship, 0=not |
# MAGIC | 20 | age_at_enrollment | Age at time of enrollment |
# MAGIC | 21 | international | 1=international student, 0=domestic |
# MAGIC | 22-27 | curricular_units_1st_sem_* | Semester 1 academic metrics |
# MAGIC | 28-33 | curricular_units_2nd_sem_* | Semester 2 academic metrics |
# MAGIC | 34 | unemployment_rate | Regional unemployment rate (%) |
# MAGIC | 35 | inflation_rate | National inflation rate (%) |
# MAGIC | 36 | gdp | National GDP growth rate (%) |
# MAGIC | 37 | target | Student outcome: "Dropout", "Enrolled", or "Graduate" |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7 Class Distribution

# COMMAND ----------

print("Target variable distribution:")
print("=" * 40)
df_bronze.groupBy("target").count().orderBy("count", ascending=False).show()

total = df_bronze.count()
dist = df_bronze.groupBy("target").count().collect()
print("Percentages:")
for row in dist:
    pct = (row["count"] / total) * 100
    print(f"  {row['target']:12s}: {row['count']:5d} ({pct:.1f}%)")
print(f"\n  Total: {total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.8 Descriptive Statistics

# COMMAND ----------

display(df_bronze.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.9 Write Bronze Delta Table
# MAGIC
# MAGIC Writing raw data as-is to `bronze.uci_dropout`. **Zero transformations applied.**

# COMMAND ----------

df_bronze.write.format("delta").mode("overwrite").saveAsTable("bronze.uci_dropout")

verify_df = spark.table("bronze.uci_dropout")
print("✅ Bronze table written successfully: bronze.uci_dropout")
print(f"   Rows: {verify_df.count()}")
print(f"   Columns: {len(verify_df.columns)}")
print(f"   Format: Delta")
print(f"   Mode: overwrite (idempotent)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Bronze Layer Complete
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - `bronze.uci_dropout` Delta table with all original columns intact
# MAGIC - Schema documented with types, null counts, and plain-English meanings
# MAGIC - Class distribution confirmed (~32% Dropout, ~18% Enrolled, ~50% Graduate)
# MAGIC - Zero transformations applied — Bronze is raw and immutable
# MAGIC
# MAGIC **Next:** Run `02_silver_layer` to clean data and engineer features.
