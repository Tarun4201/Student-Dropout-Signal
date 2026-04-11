# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 01 — Bronze Layer
# MAGIC ## The Dropout Signal: Raw Data Ingest
# MAGIC
# MAGIC **Purpose:** Ingest the UCI Dropout CSV into a governed Bronze Delta table with a documented schema.
# MAGIC
# MAGIC **Output Table:** `bronze.uci_dropout`
# MAGIC
# MAGIC **Rules:** Zero transformations — Bronze is always raw and immutable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Create Schemas
# MAGIC Create all four schemas used across the pipeline.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
spark.sql("CREATE SCHEMA IF NOT EXISTS audit")

print("✅ All schemas created: bronze, silver, gold, audit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Ingest CSV
# MAGIC
# MAGIC **IMPORTANT:** Upload `students_dropout_academic_success.csv` to DBFS before running this cell.
# MAGIC
# MAGIC Use the Databricks UI: **Data** → **Add Data** → **Upload File** → drag and drop the CSV.
# MAGIC
# MAGIC Default upload path: `/FileStore/tables/students_dropout_academic_success.csv`
# MAGIC
# MAGIC If your file is uploaded to a different path, update `CSV_PATH` below.

# COMMAND ----------

# ==========================================
# CONFIGURATION — UPDATE THIS PATH IF NEEDED
# ==========================================
CSV_PATH = "/FileStore/tables/students_dropout_academic_success.csv"

# Read the CSV with schema inference — no transformations
df_bronze = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

row_count = df_bronze.count()
col_count = len(df_bronze.columns)
print(f"✅ CSV loaded successfully")
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
# MAGIC Automated documentation: column name, data type, null count, and sample value for every column.

# COMMAND ----------

from pyspark.sql import functions as F

print(f"{'#':<4} {'Column Name':<55} {'Type':<14} {'Nulls':<8} {'Sample Value'}")
print("=" * 120)

for idx, col_name in enumerate(df_bronze.columns, 1):
    col_type = str(df_bronze.schema[col_name].dataType).replace("Type", "")
    stats = df_bronze.agg(
        F.count(F.when(F.col(f"`{col_name}`").isNull(), 1)).alias("nulls"),
        F.first(F.col(f"`{col_name}`")).alias("sample")
    ).collect()[0]
    print(f"{idx:<4} {col_name:<55} {col_type:<14} {stats['nulls']:<8} {stats['sample']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 Column Reference (Plain-English Meanings)
# MAGIC
# MAGIC | # | Column | Meaning |
# MAGIC |---|--------|---------|
# MAGIC | 1 | Marital Status | Marital status code (1=single, 2=married, 3=widowed, 4=divorced, 5=facto union, 6=legally separated) |
# MAGIC | 2 | Application mode | Method of application (e.g. 1=general, 17=post-secondary, 39=over 23 years old, etc.) |
# MAGIC | 3 | Application order | Preference order of this course in the application (1=first choice through 9) |
# MAGIC | 4 | Course | Course code identifier |
# MAGIC | 5 | Daytime/evening attendance | 1=daytime, 0=evening |
# MAGIC | 6 | Previous qualification | Code for previous qualification type |
# MAGIC | 7 | Previous qualification (grade) | Grade from previous qualification (0-200 scale) |
# MAGIC | 8 | Nacionality | Nationality code (1=Portuguese) |
# MAGIC | 9 | Mother's qualification | Mother's education level code |
# MAGIC | 10 | Father's qualification | Father's education level code |
# MAGIC | 11 | Mother's occupation | Mother's occupation code |
# MAGIC | 12 | Father's occupation | Father's occupation code |
# MAGIC | 13 | Admission grade | Admission exam grade (0-200 scale) |
# MAGIC | 14 | Displaced | 1=student is displaced from home region, 0=not |
# MAGIC | 15 | Educational special needs | 1=has special needs, 0=not |
# MAGIC | 16 | Debtor | 1=has outstanding debt, 0=not |
# MAGIC | 17 | Tuition fees up to date | 1=fees are current, 0=fees overdue |
# MAGIC | 18 | Gender | 1=male, 0=female |
# MAGIC | 19 | Scholarship holder | 1=receives scholarship, 0=does not |
# MAGIC | 20 | Age at enrollment | Age at time of enrollment |
# MAGIC | 21 | International | 1=international student, 0=domestic |
# MAGIC | 22 | Curricular units 1st sem (credited) | Number of credited units in semester 1 |
# MAGIC | 23 | Curricular units 1st sem (enrolled) | Number of enrolled units in semester 1 |
# MAGIC | 24 | Curricular units 1st sem (evaluations) | Number of evaluations in semester 1 |
# MAGIC | 25 | Curricular units 1st sem (approved) | Number of approved units in semester 1 |
# MAGIC | 26 | Curricular units 1st sem (grade) | Average grade in semester 1 (0-20 scale) |
# MAGIC | 27 | Curricular units 1st sem (without evaluations) | Units without evaluations in semester 1 |
# MAGIC | 28 | Curricular units 2nd sem (credited) | Number of credited units in semester 2 |
# MAGIC | 29 | Curricular units 2nd sem (enrolled) | Number of enrolled units in semester 2 |
# MAGIC | 30 | Curricular units 2nd sem (evaluations) | Number of evaluations in semester 2 |
# MAGIC | 31 | Curricular units 2nd sem (approved) | Number of approved units in semester 2 |
# MAGIC | 32 | Curricular units 2nd sem (grade) | Average grade in semester 2 (0-20 scale) |
# MAGIC | 33 | Curricular units 2nd sem (without evaluations) | Units without evaluations in semester 2 |
# MAGIC | 34 | Unemployment rate | Regional unemployment rate (%) |
# MAGIC | 35 | Inflation rate | National inflation rate (%) |
# MAGIC | 36 | GDP | National GDP growth rate (%) |
# MAGIC | 37 | target | Student outcome: "Dropout", "Enrolled", or "Graduate" |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7 Class Distribution

# COMMAND ----------

print("Target variable distribution:")
print("=" * 40)
df_bronze.groupBy("`target`").count().orderBy("count", ascending=False).show()

# Compute percentages
total = df_bronze.count()
dist = df_bronze.groupBy("`target`").count().collect()
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

# Verify the write
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
# MAGIC - `bronze.uci_dropout` Delta table with all 37 original columns + target
# MAGIC - Schema documented with types, null counts, and plain-English meanings
# MAGIC - Class distribution confirmed (~32% Dropout, ~18% Enrolled, ~50% Graduate)
# MAGIC - Zero transformations applied — Bronze is raw and immutable
# MAGIC
# MAGIC **Next:** Run `02_silver_layer` to clean data and engineer features.
