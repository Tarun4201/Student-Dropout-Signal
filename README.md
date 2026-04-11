# The Dropout Signal

### Fair Early-Warning Pipeline for Student Dropout Prediction

**Stack:** Databricks · Delta Lake · MLflow · SHAP · sklearn  
**Dataset:** UCI Dropout (4,424 records, 37 features)  
**Version:** 2.0 | April 2026

---

## Quick Start

### Prerequisites

- Databricks workspace with a running cluster
- Python 3.9+ (included with Databricks Runtime)
- Libraries: `xgboost`, `shap` (installed via `%pip` in notebooks)

### Step 1: Upload the Dataset

1. Open your Databricks workspace
2. Navigate to **Data** → **Add Data** → **Upload File**
3. Upload `students_dropout_academic_success.csv`
4. Note the DBFS path (default: `/FileStore/tables/students_dropout_academic_success.csv`)

### Step 2: Import Notebooks

1. Navigate to **Workspace** → your folder
2. Right-click → **Import**
3. Import all 6 `.py` files from the `notebooks/` directory

### Step 3: Run the Pipeline

Run the notebooks **in order** — each depends on the previous:

| Order | Notebook | Time | What it does |
|-------|----------|------|-------------|
| 1 | `01_bronze_layer` | ~2 min | Raw CSV → Delta table, schema docs |
| 2 | `02_silver_layer` | ~3 min | Cleaning, features, target encoding |
| 3 | `03_model_training` | ~5 min | LogReg + XGBoost + ★calibration + MLflow |
| 4 | `04_fairness_audit` | ~3 min | Marginal + ★intersectional fairness |
| 5 | `05_shap_explainability` | ~3 min | SHAP TreeExplainer, per-student top-3 |
| 6 | `06_gold_table` | ~2 min | Final output with ★reason_text + tiers |

**Total end-to-end runtime: ~18 minutes**

---

## Pipeline Architecture

```
Bronze ──→ Silver ──→ Model Training ──→ Fairness Audit
                              │                  │
                              ▼                  ▼
                      SHAP Analysis      audit.fairness_metrics
                              │
                              ▼
                        Gold Table
                  (at_risk_students)
```

### Medallion Architecture Tables

| Layer | Table | Description |
|-------|-------|-------------|
| Bronze | `bronze.uci_dropout` | Raw CSV ingest — zero transformations |
| Silver | `silver.uci_dropout_clean` | Cleaned, features engineered, target encoded |
| Silver | `silver.model_test_results` | Test set with predictions (intermediate) |
| Silver | `silver.shap_results` | Per-student SHAP top-3 (intermediate) |
| Audit | `audit.fairness_metrics` | Marginal + intersectional parity metrics |
| Gold | `gold.at_risk_students` | Final at-risk students with reason_text |

### MLflow Assets

| Asset | Name |
|-------|------|
| Experiment | `dropout_signal_hackathon` |
| Run 1 | `logistic_regression_baseline` |
| Run 2 | `xgboost_classifier` |
| Registered Model | `dropout_risk_champion` (Production, calibrated) |

---

## ★ Three Differentiators

### 1. Platt Calibration
Raw XGBoost probabilities → true calibrated probabilities via `CalibratedClassifierCV`.
Validated with a reliability diagram. Makes intervention thresholds statistically meaningful.

### 2. Intersectional Fairness Audit
Gender × socioeconomic cross-product (4 groups). Detects disparities invisible to marginal audits.
The audit most other teams skip entirely.

### 3. `reason_text` Column
Plain-English sentences for advisors built from each student's top-3 SHAP factors.
Example: *"Grade fell 2.3pts semester-on-semester; financial stress score 4/5; 67% unit non-completion rate."*

---

## Engineered Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `grade_delta` | sem2_grade − sem1_grade | Negative = declining performance |
| `absenteeism_trend` | (enr1−app1+enr2−app2)/(enr1+enr2+1) | Non-completion rate |
| `financial_stress_index` | debtor×2 + (1−fees_ok)×2 + (1−scholarship) | Range 0–5, financial risk |
| `engagement_score` | (app1/(enr1+1))+(app2/(enr2+1))+(eval1+eval2)/20 | Composite engagement |

---

## Gold Table Schema

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | int | Row index from original dataset |
| `risk_score` | float | Calibrated probability (0.0–1.0) ★ |
| `dropout_predicted` | int | Binary prediction (1=at-risk) |
| `intervention_tier` | string | "high" (≥0.70), "medium" (0.40–0.70), "low" (<0.40) |
| `shap_factor_1/2/3` | string | Top SHAP feature names |
| `shap_value_1/2/3` | float | SHAP values |
| `reason_text` | string | Plain-English advisory sentence ★ |
| `gender` | int | From Silver table |
| `financial_stress_index` | float | From Silver table |
| `grade_delta` | float | From Silver table |
| `scored_at` | timestamp | When Gold table was generated |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CSV path not found | Update `CSV_PATH` in notebook 01 to match your DBFS upload path |
| `xgboost` import error | Ensure `%pip install xgboost` ran successfully in notebook 03 |
| `shap` import error | Ensure `%pip install shap` ran successfully in notebook 05 |
| Unity Catalog errors | Tables use simple `schema.table` naming — no catalog prefix needed |
| MLflow experiment not found | Notebooks 04-06 auto-search for the experiment — run notebook 03 first |

---

## Team

Built for the 24-hour hackathon using the UCI Dropout Dataset.

**The Dropout Signal — v2.0 | April 2026**
