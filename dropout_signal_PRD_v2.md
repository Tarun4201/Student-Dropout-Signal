# PRODUCT REQUIREMENTS DOCUMENT
## The Dropout Signal
### Fair Early-Warning Pipeline for Student Dropout Prediction

**24-Hour Hackathon · Team Size: 3–4 · Dataset: UCI Dropout (4,400 records, 37 features)**  
**Stack:** Databricks · Delta Lake · MLflow · Unity Catalog · SHAP · sklearn.calibration  
**Version 2.0 | April 2026**

---

> ★ **DIFFERENTIATOR — Three competitive differentiators in v2.0**
>
> This version of the PRD integrates three additions that separate The Dropout Signal from every other team's pipeline: **(1) Platt calibration** — so risk scores are statistically meaningful, not raw model outputs; **(2) Intersectional fairness audit** — detecting disparities invisible to marginal audits; **(3) `reason_text` column in the Gold table** — plain-English sentences for advisors who don't know what SHAP is. Each differentiator is marked ★ throughout this document.

---

## 1. Executive Summary

This PRD defines the complete build plan for **The Dropout Signal** — a fair, explainable, end-to-end machine learning pipeline that predicts student dropout risk using the UCI Dropout Dataset. The system is designed to win across all five judging criteria within a 24-hour hackathon window.

The core output is a Gold-layer Delta table of at-risk students with calibrated risk scores, top three contributing SHAP risk factors per student, a plain-English `reason_text` sentence, and a tiered intervention recommendation (low / medium / high). A documented fairness audit — including intersectional analysis across gender and socioeconomic group — ensures the pipeline does not discriminate against students by any protected attribute or their intersection.

### 1.1 Why This Problem Statement

| Factor | Assessment |
|--------|------------|
| Dataset size | 4,400 rows — trains in seconds, no cluster tuning needed |
| Feature richness | 37 features covering grades, finance, demographics, macroeconomics |
| Fairness (25% weight) | Highest-weighted criterion — most teams skip it, giving you an edge |
| Pipeline clarity | Bronze → Silver → Gold maps cleanly to 3 time blocks |
| Explainability | SHAP works out-of-the-box on XGBoost with `pip install shap` |
| Risk level | Low — no multi-table joins, no LLM hallucination pipeline, no PSI math |

### 1.2 Scoring Strategy

| Criterion | Weight | Target | Key actions |
|-----------|--------|--------|-------------|
| Pipeline architecture | 20% | Full marks | Strict Bronze/Silver/Gold naming, Delta format throughout, Unity Catalog registration |
| Feature engineering | 20% | Full marks | 3 required features + bonus `engagement_score` + `reason_text` generation — all documented |
| Model + MLflow hygiene | 20% | Full marks | Named runs, all params/metrics logged, calibration curve artifact, champion registered |
| Fairness audit | 25% | Full marks | Marginal AND intersectional parity/EOD computed, findings honest, disparity sourced and mitigated |
| Explainability + Gold table | 15% | Full marks | SHAP top-3 per student, calibrated risk score, `reason_text` plain-English column, defensible tiers |

---

## 2. Problem Context & Goals

### 2.1 Problem Statement

Universities lose students to dropout every year, but the warning signals appear months earlier. Declining semester grades, missed financial obligations, and disengagement are all measurable before the decision to leave is made. The challenge is not just building a predictive model — it is building a **fair** one.

A model that disproportionately flags students from lower-income backgrounds or specific demographic groups is a liability, not a tool. The pipeline must predict risk early, explain the reasoning transparently per student, deliver that reasoning in plain English to non-technical advisors, and demonstrate that it does not encode bias against any protected group or their intersection.

### 2.2 Primary Goals

1. Ingest the UCI Dropout dataset into a governed Bronze Delta table with a documented schema.
2. Clean and engineer features in a Silver Delta table, including the three mandatory features, the bonus `engagement_score`, and the `reason_text` generation logic.
3. Train a Logistic Regression baseline and an XGBoost model. Log both in MLflow. Apply Platt calibration to the champion model. Register the calibrated model in Model Registry. ★
4. Conduct a fairness audit across gender and socioeconomic group — both marginal and intersectional. Compute demographic parity difference and equal opportunity difference for all combinations. Log results as a Delta table. Document all findings honestly. ★
5. Run SHAP on the best model to surface the top three risk factors per flagged student.
6. Produce a Gold output table with: student ID, calibrated risk score, `reason_text` plain-English column, top three SHAP factors, and a recommended intervention tier. ★

### 2.3 Out of Scope (within 24 hours)

- OULAD multi-table enrichment — only attempted after hour 21 if the core pipeline is complete and stable.
- Real-time serving or REST API endpoint.
- UI dashboard or visualisation layer.
- Hyperparameter tuning beyond a reasonable grid — diminishing returns on 4,400 rows.
- World Bank or AISHE enrichment data.
- Survival/time-to-event model (noted as bonus extension only — not required for full marks).

---

## 3. Dataset Specification

### 3.1 Primary Dataset — UCI Dropout

| Property | Detail |
|----------|--------|
| Source URL | https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success |
| File name | `uci_dropout.csv` |
| Records | 4,424 rows |
| Features | 37 columns |
| Target column | `Target` (string: "Dropout", "Enrolled", "Graduate") |
| Binary target | Dropout = 1, Enrolled or Graduate = 0 |
| Auth | None — direct access, no Kaggle login required |

### 3.2 Target Variable

- **Original column:** `Target` (values: "Dropout", "Enrolled", "Graduate")
- **Binary encoding:** `dropout_label = 1` where `Target == "Dropout"`, else `0`
- **Class distribution (approx):** ~32% Dropout, ~18% Enrolled, ~50% Graduate
- **Implication:** mild class imbalance — use `class_weight="balanced"` in LogReg, `scale_pos_weight` in XGBoost
- **Do NOT** use Enrolled/Graduate distinction — problem statement specifies binary classification only

---

## 4. System Architecture

### 4.1 Medallion Architecture Overview

| Layer | Table name | Description |
|-------|------------|-------------|
| Bronze | `bronze.uci_dropout` | Raw ingest of `uci_dropout.csv` — no transformations, schema documented |
| Silver | `silver.uci_dropout_clean` | Null handling, binary target encoding, all engineered features including `reason_text` template ★ |
| Gold | `gold.at_risk_students` | At-risk students with calibrated risk score ★, `reason_text` ★, SHAP top-3, intervention tier |
| Audit | `audit.fairness_metrics` | Marginal and intersectional ★ parity and equal opportunity scores by group and model run |

### 4.2 MLflow Experiment Structure

| MLflow object | Specification |
|---------------|---------------|
| Experiment name | `dropout_signal_hackathon` |
| Run 1 name | `logistic_regression_baseline` |
| Run 2 name | `xgboost_classifier` |
| Logged params (LR) | `C`, `max_iter`, `class_weight`, `solver` |
| Logged params (XGB) | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |
| Logged metrics (both) | `accuracy`, `roc_auc`, `f1_macro`, `precision`, `recall` |
| Logged artifacts | `confusion_matrix.png`, `feature_importance.csv`, `shap_summary.png`, `reliability_diagram.png` ★ |
| Model Registry name | `dropout_risk_champion` |
| Registered model stage | Production (calibrated model) |

### 4.3 Unity Catalog Registration

| Asset | Unity Catalog path |
|-------|--------------------|
| Bronze table | `main.bronze.uci_dropout` |
| Silver table | `main.silver.uci_dropout_clean` |
| Gold table | `main.gold.at_risk_students` |
| Fairness audit table | `main.audit.fairness_metrics` |
| Champion model | MLflow Model Registry: `dropout_risk_champion` (calibrated) ★ |

---

## 5. Pipeline Specifications

### 5.1 Step 1 — Bronze Layer (Hours 0–2)

**Deliverable:** `bronze.uci_dropout` — raw Delta table, schema documented, no transformations applied

#### 5.1.1 Tasks

1. Download `uci_dropout.csv` from the Kaggle URL or confirm path if pre-loaded in cluster.
2. Ingest into Databricks using `spark.read.csv()` with `inferSchema=True` and `header=True`.
3. Write as Delta format to `bronze.uci_dropout`. Mode: `overwrite`.
4. Run `df.printSchema()` and `df.describe()` — save output to a Markdown cell.
5. Document every column: name, type, sample values, null count, and plain-English meaning.
6. Confirm class distribution: `df.groupBy("Target").count().show()`.

#### 5.1.2 Acceptance criteria

- Delta table exists at `bronze.uci_dropout` with all 37 original columns intact.
- Schema documentation cell is present in the notebook.
- Zero transformation applied — Bronze is always raw and immutable.
- Null counts documented for every column.

---

### 5.2 Step 2 — Silver Layer (Hours 2–5)

**Deliverable:** `silver.uci_dropout_clean` — cleaned Delta table with binary target + all engineered features

#### 5.2.1 Null handling

- For numeric columns: impute with column **median** (not mean — more robust to grade outliers).
- For categorical columns: impute with mode or a dedicated "Unknown" category.
- Document the imputation strategy and null counts before/after in a notebook cell.
- Do not drop rows — 4,400 records is small; every row is valuable.

#### 5.2.2 Target encoding

- Create new column `dropout_label`: `1` where `Target == "Dropout"`, else `0`.
- Retain original `Target` column in Silver — do not drop it.
- Verify balance: print class counts and ratio.

#### 5.2.3 Engineered features

| Feature column | Formula | Interpretation / notes |
|----------------|---------|------------------------|
| `grade_delta` | `sem2_grade − sem1_grade` | Negative = declining trend; strong dropout signal |
| `absenteeism_trend` | `(enr1−app1+enr2−app2)/(enr1+enr2+1)` | Rate of enrolling but not completing; proxy for disengagement |
| `financial_stress_index` | `debtor×2 + (1−fees_ok)×2 + (1−scholarship)` | Range 0–5; higher weight on debt/overdue fees |
| `engagement_score` | `(app1/(enr1+1))+(app2/(enr2+1))+(eval1+eval2)/20` | Composite unit-completion + assessment participation |
| `reason_text` ★ | Generated from `shap_factor_1/2/3` values | Plain-English sentence for advisors; e.g. "Grade fell 2.3pts; 67% non-completion; debt outstanding." |

> ★ **DIFFERENTIATOR — `reason_text` — plain-English advisor column**
>
> After SHAP values are computed (Step 5), generate a `reason_text` string from the top-3 SHAP factors for each at-risk student using the `factor_interpretations` dict in Section 8.6. This column is the primary deliverable that advisors will read — it translates model output into human language without requiring the reader to understand SHAP.

#### 5.2.4 Acceptance criteria

- All nulls handled — `df.isnull().sum()` returns zero.
- `dropout_label` column exists with values `0` and `1` only.
- `grade_delta`, `absenteeism_trend`, `financial_stress_index`, `engagement_score` all present with documented formulas.
- Silver Delta table written to `silver.uci_dropout_clean`.

---

### 5.3 Step 3 — Model Training & MLflow (Hours 5–9)

**Deliverable:** Two trained models logged in MLflow. Calibrated champion model registered as `"dropout_risk_champion"` in Model Registry.

#### 5.3.1 Train/test split

- Split: 80% train, 20% test. Random state: `42` (reproducible).
- Stratify on `dropout_label` to preserve class balance in both splits.
- Features: all Silver columns except `Target`, `dropout_label`, and raw ID columns.
- Log the train/test sizes as MLflow params.

#### 5.3.2 Model 1 — Logistic Regression baseline

| Parameter | Value |
|-----------|-------|
| Class | `sklearn.linear_model.LogisticRegression` |
| C (regularisation) | `1.0` (log to MLflow) |
| `max_iter` | `1000` |
| `class_weight` | `"balanced"` — essential given ~32% dropout class |
| `solver` | `"lbfgs"` |
| MLflow run name | `logistic_regression_baseline` |

#### 5.3.3 Model 2 — XGBoost classifier

| Parameter | Value |
|-----------|-------|
| Class | `xgboost.XGBClassifier` |
| `n_estimators` | `200` |
| `max_depth` | `5` |
| `learning_rate` | `0.1` |
| `scale_pos_weight` | ratio of negative to positive class (auto-compute) |
| `eval_metric` | `"auc"` |
| MLflow run name | `xgboost_classifier` |

#### 5.3.4 ★ Model Calibration (Hours 9–10)

> ★ **DIFFERENTIATOR — Platt calibration — scores become true probabilities**
>
> Raw XGBoost `predict_proba` output is not a calibrated probability. A score of 0.7 does not mean a 70% dropout chance. Apply Platt scaling via `CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")`. Fit on a validation split (not the test set). Log a `reliability_diagram.png` to MLflow. Register the calibrated model — not the raw one — as `dropout_risk_champion`. This is what makes your intervention tier thresholds statistically defensible.

- **Install:** `CalibratedClassifierCV` is part of `sklearn.calibration` — no new dependency.
- Fit on a held-out validation split carved from the training set (e.g. 10% of train).
- Generate `calibration_curve()` and save as `reliability_diagram.png` — log to MLflow via `mlflow.log_artifact()`.
- All downstream `risk_score` values in the Gold table must use `calibrated_model.predict_proba()`, not `xgb_model.predict_proba()`.
- Log to MLflow: `calibration_method="platt_scaling"`, `val_size` as param.

#### 5.3.5 Metrics to log for both models

- `roc_auc` — primary comparison metric
- `f1_macro` — accounts for class imbalance
- `accuracy` — for completeness
- `precision` and `recall` — important for fairness discussion
- Log confusion matrix as a `.png` artifact using `mlflow.log_artifact()`.
- ★ Log `reliability_diagram.png` after calibration (champion model only).

#### 5.3.6 Model Registry

- Register the calibrated champion model under name `dropout_risk_champion`.
- Set stage to **Production**.
- Tag with: `dataset=uci_dropout`, `version=2.0`, `fairness_audited=pending`, `calibrated=true`.
- After fairness audit, update tag `fairness_audited=true`.

#### 5.3.7 Acceptance criteria

- Both runs visible in MLflow UI under experiment `dropout_signal_hackathon`.
- All params and metrics logged — no empty fields.
- `reliability_diagram.png` artifact present in the champion run.
- Calibrated model (not raw XGBoost) registered at Production stage.

---

### 5.4 Step 4 — Fairness Audit (Hours 9–15)

**Deliverable:** `audit.fairness_metrics` Delta table with marginal and intersectional parity metrics. Written 1-page findings in notebook.

#### 5.4.1 Why this step wins the hackathon

The fairness audit carries **25% of the total score** — the single largest criterion. Most teams either skip it entirely or compute one marginal metric and move on. A thorough audit including intersectional analysis, with honest documentation of any disparities found, is the highest-leverage work you can do in this hackathon. Judges explicitly reward teams that do not hide their findings.

#### 5.4.2 Fairness groupings

| Group | How to derive |
|-------|---------------|
| Gender | Column `"Gender"` — values `0` and `1` (binary split) |
| Socioeconomic group | Derive from `financial_stress_index`: score ≥ 3 = `"high_stress"`, score < 3 = `"low_stress"` |
| Scholarship status | Column `"Scholarship_holder"` — secondary grouping |
| ★ Intersectional (new) | 4-way: `female_high_stress` / `female_low_stress` / `male_high_stress` / `male_low_stress` — computed from `Gender × socioeconomic_group` cross-product |

> ★ **DIFFERENTIATOR — Intersectional fairness — the audit most teams skip entirely**
>
> A model can pass a marginal gender audit and a marginal socioeconomic audit while still systematically under-serving female students in financial hardship — the most vulnerable intersection. Compute all four gender × socioeconomic combinations: `female_high_stress`, `female_low_stress`, `male_high_stress`, `male_low_stress`. Compare each pair. The expected finding is that `female_high_stress` has the largest equal opportunity difference — document it honestly, name the likely cause (`financial_stress_index` correlates with both group membership and dropout), and state the production mitigation. **This is the slide that wins the 25%.**

#### 5.4.3 Fairness metrics to compute

**Metric 1 — Demographic Parity Difference**

- **Definition:** `|P(predicted=1 | group=A) − P(predicted=1 | group=B)|`
- **Measures:** are positive predictions (at-risk flags) equally distributed across groups?
- **Threshold:** difference > 0.10 is considered significant and must be documented
- **Compute for:** Gender (marginal), Socioeconomic (marginal), all intersectional pairs ★

**Metric 2 — Equal Opportunity Difference**

- **Definition:** `|TPR(group=A) − TPR(group=B)|` where TPR = true positive rate (recall)
- **Measures:** does the model correctly catch actual dropouts equally across groups?
- **Threshold:** difference > 0.10 is considered significant
- **Compute for:** Gender (marginal), Socioeconomic (marginal), all intersectional pairs ★
- This metric matters more ethically — it measures whether the model fails to help at-risk students equally

#### 5.4.4 Fairness Delta table schema

| Column | Type | Example value / notes |
|--------|------|-----------------------|
| `run_id` | String | MLflow run ID of evaluated model |
| `model_name` | String | `"xgboost_classifier"` |
| `group_type` | String | `"gender"`, `"socioeconomic"`, or `"intersectional_gender_x_socioeconomic"` ★ |
| `group_a` | String | `"male"`, `"high_stress"`, or `"female_high_stress"` ★ |
| `group_b` | String | `"female"`, `"low_stress"`, or `"male_low_stress"` ★ |
| `audit_type` | String ★ | `"marginal"` or `"intersectional"` — new column in v2.0 |
| `demographic_parity_diff` | Float | `0.07` |
| `equal_opportunity_diff` | Float | `0.13` |
| `group_a_positive_rate` | Float | `0.38` |
| `group_b_positive_rate` | Float | `0.31` |
| `group_a_tpr` | Float | `0.71` |
| `group_b_tpr` | Float | `0.58` |
| `evaluated_at` | Timestamp | `2026-04-12 09:30:00` |

#### 5.4.5 Findings documentation

In a dedicated notebook cell, write a plain-English findings section covering:

- Whether demographic parity difference exceeded 0.10 for any marginal group combination — and if so, by how much.
- Whether equal opportunity difference exceeded 0.10 for any marginal group — and which group is being under-served.
- ★ Which intersectional group has the largest equal opportunity difference — name the specific intersection (e.g. `female_high_stress` vs `male_low_stress`).
- ★ A hypothesis for why the intersectional disparity exists and why marginal audits would have missed it.
- A statement on what would be done to mitigate the disparity in production (e.g. reweighting, fairness constraints, or separate thresholds).

> **Do not sanitise or hide findings.** Judges penalise teams that present a clean audit when the dataset is known to contain socioeconomic signals correlated with dropout.

#### 5.4.6 Acceptance criteria

- `audit.fairness_metrics` Delta table exists with at least 8 rows: 2 metrics × 2 marginal groups + 2 metrics × up to 6 intersectional pairs.
- `audit_type` column contains both `"marginal"` and `"intersectional"` values.
- Both demographic parity difference and equal opportunity difference computed for Gender, Socioeconomic, and all intersectional combinations.
- Written findings section present in notebook — minimum 200 words.
- MLflow model tag updated: `fairness_audited=true`.

---

### 5.5 Step 5 — SHAP Explainability (Hours 14–18)

**Deliverable:** SHAP summary plot logged to MLflow. Top-3 SHAP factors per flagged student added to Gold table.

#### 5.5.1 Setup

- **Install:** `pip install shap`
- Use `shap.TreeExplainer` for XGBoost — faster than `KernelExplainer`, exact for tree models.
- Compute SHAP values only on the test set to avoid data leakage in explanations.

#### 5.5.2 Global explainability

- Generate `shap.summary_plot()` — bar chart of mean absolute SHAP values across all features.
- Save as `shap_summary.png` and log via `mlflow.log_artifact()`.
- Identify and document the top 5 global predictors — these should appear in the presentation.

#### 5.5.3 Per-student explainability

- For each student predicted as at-risk, extract their individual SHAP values.
- Identify the three features with the highest absolute SHAP values for that student.
- Store as: `shap_factor_1/2/3` (feature names) and `shap_value_1/2/3` (float values).
- ★ Pass these immediately to the `reason_text` generator (Section 8.6) before writing the Gold table.

#### 5.5.4 Acceptance criteria

- `shap_summary.png` artifact logged in MLflow run.
- Every at-risk student in the Gold table has non-null values in `shap_factor_1/2/3`.
- ★ Every at-risk student in the Gold table has a non-null, non-generic `reason_text` value.
- SHAP computed using `TreeExplainer` (not `KernelExplainer` — too slow).

---

### 5.6 Step 6 — Gold Output Table (Hours 16–18)

**Deliverable:** `gold.at_risk_students` — final consumable output registered in Unity Catalog

#### 5.6.1 Gold table schema

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `student_id` | Integer | No | Row index from original dataset (synthetic ID) |
| `risk_score` | Float | No | Calibrated probability (0.0–1.0) — Platt-scaled, not raw XGBoost output ★ |
| `dropout_predicted` | Integer | No | Binary prediction: 1 = at-risk, 0 = not at-risk |
| `intervention_tier` | String | No | `"high"` (≥0.70), `"medium"` (0.40–0.70), `"low"` (<0.40) |
| `shap_factor_1` | String | No | Name of top contributing risk feature |
| `shap_value_1` | Float | No | SHAP value for factor 1 |
| `shap_factor_2` | String | No | Name of second contributing risk feature |
| `shap_value_2` | Float | No | SHAP value for factor 2 |
| `shap_factor_3` | String | No | Name of third contributing risk feature |
| `shap_value_3` | Float | No | SHAP value for factor 3 |
| `reason_text` ★ | String | No | Plain-English sentence built from SHAP factors for advisors — e.g. "Grade fell 2.3pts; 67% non-completion; debt outstanding." |
| `gender` | Integer | No | From Silver — for downstream fairness reanalysis |
| `financial_stress_index` | Float | No | From Silver — for downstream filtering |
| `grade_delta` | Float | No | From Silver — for downstream filtering |
| `scored_at` | Timestamp | No | Timestamp when Gold table was generated |

#### 5.6.2 Intervention tier logic

| Tier | Assignment rule |
|------|-----------------|
| **HIGH** | `risk_score ≥ 0.70` — immediate outreach by academic advisor recommended |
| **MEDIUM** | `risk_score ≥ 0.40` and `< 0.70` — scheduled check-in within 2 weeks |
| **LOW** | `risk_score < 0.40` — monitor only, no active intervention |

**Rationale:** thresholds apply to calibrated risk scores. At ~32% base rate, a calibrated score of 0.70 implies roughly 2.2× higher dropout probability than base. Because scores are Platt-calibrated, the 0.70 threshold carries statistical meaning: it approximates the score at which precision reaches an acceptable level for advisor time investment. Document this rationale in the notebook.

#### 5.6.3 Acceptance criteria

- Gold table contains only students with `dropout_predicted == 1`.
- Zero null values in `student_id`, `risk_score`, `intervention_tier`, `shap_factor` columns, and `reason_text`.
- `risk_score` values are calibrated probabilities — derived from `calibrated_model.predict_proba()`, not raw XGBoost. ★
- `reason_text` contains human-readable sentences referencing actual student values (not feature names alone). ★
- Table registered in Unity Catalog at `main.gold.at_risk_students`.
- `intervention_tier` values are only `"high"`, `"medium"`, or `"low"` — no nulls, no other strings.

---

## 6. Execution Timeline

### 6.1 Hour-by-Hour Schedule

| Hours | Phase | Owner | Key outputs |
|-------|-------|-------|-------------|
| 0–2 | Bronze layer + setup | Person 1 | `bronze.uci_dropout`, schema docs, class distribution confirmed |
| 2–5 | Silver layer + features | Persons 1 & 2 | `silver.uci_dropout_clean`, `grade_delta`, `absenteeism_trend`, `financial_stress_index`, `reason_text` template |
| 5–7 | EDA + train/test split | Person 2 | Distribution plots, correlation analysis, split documented in MLflow |
| 7–9 | Model training + MLflow | Person 3 | Both models trained, all metrics logged, champion registered |
| 9–10 | ★ Model calibration | Person 3 | `CalibratedClassifierCV`, `reliability_diagram.png` logged to MLflow, calibrated model registered |
| 9–14 | Fairness audit (marginal) | Person 4 | `audit.fairness_metrics` marginal rows (gender + socioeconomic), written findings |
| 14–15 | ★ Intersectional audit | Person 4 | 4 intersection groups computed, appended to `audit.fairness_metrics`, findings updated |
| 14–16 | SHAP global | Person 3 | `TreeExplainer` run, `shap_summary.png` logged to MLflow |
| 16–18 | Gold table + `reason_text` | Persons 3 & 4 | `gold.at_risk_students` complete with calibrated scores and `reason_text` column |
| 18–20 | Documentation | All | README, notebook cells clean and commented, MLflow run names verified |
| 20–22 | Presentation prep | Person 4 | 3–4 slide summary, intersectional fairness narrative, demo walkthrough |
| 22–23 | End-to-end re-run | All | Full pipeline runs clean Bronze to Gold without errors |
| 23–24 | Buffer / OULAD bonus | All | Bug fixes only; OULAD join if pipeline is stable (optional) |

### 6.2 Critical Path

- **Hour 2:** Bronze Delta table must be complete before Silver work begins.
- **Hour 5:** Silver table must be complete before model training begins.
- **Hour 9:** Models must be registered in MLflow before fairness audit and calibration begin.
- **Hour 10:** Calibrated model must be registered before SHAP and Gold table work begins. ★
- **Hour 15:** Intersectional fairness audit must be complete before SHAP and Gold table work begins. ★

### 6.3 Time Risk Management

| Risk | Mitigation |
|------|------------|
| Databricks cluster setup takes > 1 hour | Start cluster immediately at hour 0 while reading this PRD. |
| SHAP computation slow on full dataset | Run SHAP on test set only (~880 rows). `TreeExplainer` completes in under 2 minutes. |
| XGBoost import error | `pip install xgboost` at top of notebook. Fallback: `RandomForestClassifier`. |
| `fairlearn` not available | Compute fairness metrics manually — all metrics are 4-line pandas calculations. |
| ★ Calibration degrades model ranking | `CalibratedClassifierCV` with `cv="prefit"` only adjusts probabilities, not rankings — AUC is unchanged. |
| ★ Intersectional group too small | If any intersection has < 30 positives, flag the metric as "n too small" in findings. Do not omit. |
| Gold table has null SHAP or `reason_text` values | SHAP and `reason_text` generation must run before Gold table is built — block this dependency explicitly. |
| Overtime on model tuning | Hard cap model training at hour 9. Default hyperparameters yield competitive AUC on this dataset. |

---

## 7. Team Roles & Responsibilities

### 7.1 Role Assignments

#### Person 1 — Data Engineer (Hours 0–5)

- Full ownership of Bronze and Silver layers.
- Responsible for: Delta table creation, null handling, binary target encoding.
- Owns: `bronze.uci_dropout`, `silver.uci_dropout_clean`.
- Must know: PySpark DataFrame API, Delta write syntax, basic pandas.
- Hands off Silver table to Persons 2 and 3 at hour 5.

#### Person 2 — Feature Engineer (Hours 2–7)

- Feature engineering and EDA.
- Responsible for: `grade_delta`, `absenteeism_trend`, `financial_stress_index`, `engagement_score` formulas.
- ★ Also responsible for: `factor_interpretations` dict used in `reason_text` generation.
- Owns: feature definitions, correlation analysis, EDA visualisations.
- Must know: pandas, basic statistics, matplotlib or seaborn.

#### Person 3 — ML Engineer (Hours 7–18)

- Model training, MLflow, calibration, SHAP.
- Responsible for: Logistic Regression, XGBoost, MLflow logging, Model Registry, ★ Platt calibration, SHAP `TreeExplainer`.
- Owns: MLflow experiment, champion model, `reliability_diagram.png`, `shap_summary.png`, per-student SHAP values, ★ `reason_text` population.
- Must know: sklearn, xgboost, mlflow Python API, `sklearn.calibration`, basic SHAP.

#### Person 4 — Fairness Lead & Presenter (Hours 9–22)

- ★ Fairness audit (marginal + intersectional), Gold table, documentation, presentation.
- Responsible for: fairness metric computation (all groups and intersections), audit Delta table, written findings, Gold table assembly, 3-slide deck.
- Owns: `audit.fairness_metrics`, `gold.at_risk_students`, presentation narrative.
- Must know: pandas `groupby`, basic statistics, clear writing.
- This is the most judge-facing role — quality of intersectional findings write-up directly impacts the 25% fairness score.

### 7.2 Communication Checkpoints

| Time | Checkpoint |
|------|------------|
| Hour 2 | Person 1 confirms Bronze table is live — all others unblocked to proceed |
| Hour 5 | Silver table signed off — Person 3 begins model training |
| Hour 9 | Models registered in MLflow — Person 3 begins calibration, Person 4 begins marginal fairness audit |
| Hour 10 | ★ Calibrated model registered — Person 3 can begin SHAP prep |
| Hour 15 | ★ Intersectional fairness findings drafted — team reviews together |
| Hour 18 | Gold table complete with `reason_text` — team reviews sample rows together |
| Hour 20 | Full end-to-end pipeline reviewed — identify any broken steps |
| Hour 22 | Presentation walkthrough — all team members must be able to explain any slide |

---

## 8. Code Reference

### 8.1 Bronze Ingest

```python
df_bronze = spark.read.csv('/path/to/uci_dropout.csv', header=True, inferSchema=True)
df_bronze.write.format('delta').mode('overwrite').saveAsTable('bronze.uci_dropout')
df_bronze.printSchema()
df_bronze.describe().show()
df_bronze.groupBy('Target').count().show()
```

### 8.2 Silver Feature Engineering

```python
from pyspark.sql import functions as F

df_silver = df_bronze \
    .withColumn('dropout_label', F.when(F.col('Target') == 'Dropout', 1).otherwise(0)) \
    .withColumn('grade_delta', F.col('Curricular_units_2nd_sem_grade') -
        F.col('Curricular_units_1st_sem_grade')) \
    .withColumn('absenteeism_trend',
        (F.col('Curricular_units_1st_sem_enrolled') -
         F.col('Curricular_units_1st_sem_approved') +
         F.col('Curricular_units_2nd_sem_enrolled') -
         F.col('Curricular_units_2nd_sem_approved')) /
        (F.col('Curricular_units_1st_sem_enrolled') +
         F.col('Curricular_units_2nd_sem_enrolled') + 1)) \
    .withColumn('financial_stress_index',
        F.col('Debtor') * 2 + (1 - F.col('Tuition_fees_up_to_date')) * 2 + (1 -
        F.col('Scholarship_holder')))

df_silver.write.format('delta').mode('overwrite').saveAsTable('silver.uci_dropout_clean')
```

### 8.3 MLflow Logging Pattern

```python
import mlflow, mlflow.sklearn

mlflow.set_experiment('dropout_signal_hackathon')

with mlflow.start_run(run_name='xgboost_classifier') as run:
    mlflow.log_params({'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1})
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, proba))
    mlflow.log_metric('f1_macro', f1_score(y_test, preds, average='macro'))
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact('confusion_matrix.png')
```

### 8.4 ★ Platt Calibration

> ★ **DIFFERENTIATOR — Calibration code — register this model, not the raw XGBoost**
>
> Apply after model training. Fit on a validation split carved from the training data. Register the `calibrated_model` in MLflow Model Registry, not `xgb_model`.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Fit Platt scaling on held-out validation split
calibrated_model = CalibratedClassifierCV(xgb_model, cv='prefit', method='sigmoid')
calibrated_model.fit(X_val, y_val)  # X_val carved from training set, NOT X_test

# Generate reliability diagram
prob_true, prob_pred = calibration_curve(
    y_test, calibrated_model.predict_proba(X_test)[:, 1], n_bins=10)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, marker='o', label='XGBoost (calibrated)')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.legend(); fig.tight_layout()
fig.savefig('reliability_diagram.png')

with mlflow.start_run(run_id=champion_run_id):
    mlflow.log_artifact('reliability_diagram.png')
    mlflow.log_param('calibration_method', 'platt_scaling')
    mlflow.sklearn.log_model(calibrated_model, 'calibrated_model')
```

### 8.5 Fairness Metrics — Marginal

```python
results_df['socioeconomic_group'] = results_df['financial_stress_index'].apply(
    lambda x: 'high_stress' if x >= 3 else 'low_stress')

for group_col in ['gender', 'socioeconomic_group']:
    rates = results_df.groupby(group_col)['dropout_pred'].mean()
    tprs = results_df[results_df['dropout_label']==1].groupby(group_col)['dropout_pred'].mean()
    dp_diff = abs(rates.iloc[0] - rates.iloc[1])
    eo_diff = abs(tprs.iloc[0] - tprs.iloc[1])
    print(f'{group_col}: DP_diff={dp_diff:.3f}, EO_diff={eo_diff:.3f}')
```

### 8.5b ★ Fairness Metrics — Intersectional

> ★ **DIFFERENTIATOR — Intersectional audit code — append these rows to the same `audit.fairness_metrics` table**
>
> Run after the marginal audit. Computes all pairwise comparisons across 4 intersection groups.

```python
results_df['intersection'] = results_df['gender'].map({0:'female', 1:'male'}) + '_' + \
    results_df['socioeconomic_group']

intersections = results_df['intersection'].unique()
intersectional_rows = []

for i, g_a in enumerate(intersections):
    for g_b in intersections[i+1:]:
        sub_a = results_df[results_df['intersection']==g_a]
        sub_b = results_df[results_df['intersection']==g_b]
        pos_rate_a = sub_a['dropout_pred'].mean()
        pos_rate_b = sub_b['dropout_pred'].mean()
        tpr_a = sub_a[sub_a['dropout_label']==1]['dropout_pred'].mean()
        tpr_b = sub_b[sub_b['dropout_label']==1]['dropout_pred'].mean()
        intersectional_rows.append({
            'group_type': 'intersectional_gender_x_socioeconomic',
            'group_a': g_a, 'group_b': g_b,
            'audit_type': 'intersectional',
            'demographic_parity_diff': abs(pos_rate_a - pos_rate_b),
            'equal_opportunity_diff': abs(tpr_a - tpr_b),
            ...
        })
```

### 8.6 SHAP Top-3 Per Student

```python
import shap, numpy as np

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
feature_names = X_test.columns.tolist()

records = []
for i, row_shap in enumerate(shap_values):
    top3_idx = np.argsort(np.abs(row_shap))[::-1][:3]
    records.append({
        'shap_factor_1': feature_names[top3_idx[0]], 'shap_value_1': row_shap[top3_idx[0]],
        'shap_factor_2': feature_names[top3_idx[1]], 'shap_value_2': row_shap[top3_idx[1]],
        'shap_factor_3': feature_names[top3_idx[2]], 'shap_value_3': row_shap[top3_idx[2]],
    })
```

### 8.7 ★ `reason_text` Generation

> ★ **DIFFERENTIATOR — Plain-English reason sentences for advisors**
>
> Run after SHAP, before writing the Gold table. Add `reason_text` to each at-risk student record.

```python
factor_interpretations = {
    'grade_delta': lambda v: f"grade {'fell' if v<0 else 'rose'} {abs(v):.1f}pts semester-on-semester",
    'financial_stress_index': lambda v: f"financial stress score {v:.0f}/5",
    'absenteeism_trend': lambda v: f"{v*100:.0f}% unit non-completion rate",
    'Curricular_units_2nd_sem_grade': lambda v: f"semester 2 grade of {v:.1f}",
    'Debtor': lambda v: "outstanding debt on record" if v==1 else "no debt",
    'Tuition_fees_up_to_date': lambda v: "tuition fees overdue" if v==0 else "fees current",
    'Scholarship_holder': lambda v: "no scholarship" if v==0 else "scholarship holder",
}

def build_reason_text(row):
    reasons = []
    for factor_col in ['shap_factor_1', 'shap_factor_2', 'shap_factor_3']:
        feature = row[factor_col]
        value = row.get(feature, None)
        if feature in factor_interpretations and value is not None:
            reasons.append(factor_interpretations[feature](value))
        else:
            reasons.append(feature.replace('_', ' '))
    return '; '.join(reasons).capitalize() + '.'

gold_df['reason_text'] = gold_df.apply(build_reason_text, axis=1)

# Example: "Grade fell 2.3pts semester-on-semester; financial stress score 4/5; 67% unit non-completion rate."
```

---

## 9. Judging Preparation

### 9.1 Anticipated Judge Questions

| Question | Prepared answer |
|----------|-----------------|
| Why XGBoost over Random Forest? | XGBoost handles class imbalance natively via `scale_pos_weight`, trains faster on this dataset, and produces sharper SHAP values. We validated by logging the Logistic Regression baseline — XGBoost AUC was higher. |
| Why is your intervention tier threshold at 0.70 for "high"? | At ~32% base dropout rate, a calibrated score of 0.70 represents roughly 2.2× the base probability. We chose this to prioritise precision — advisors have limited time and false alarms erode trust. |
| ★ Why are your risk scores "calibrated"? | Raw XGBoost `predict_proba` output is not a true probability — a score of 0.7 does not mean 70% dropout chance. We applied Platt scaling via `CalibratedClassifierCV` and validated with a reliability diagram. Only then do the intervention tier thresholds carry a statistical meaning. |
| Did you find any fairness disparities? | Yes — we found [your actual numbers] in equal opportunity difference across socioeconomic groups, and a larger disparity at the female × high_stress intersection. We document this honestly. The disparity likely arises because `financial_stress_index` is both a strong predictor and correlated with group membership. |
| ★ What is intersectional fairness and why does it matter? | A model can be fair on gender alone and fair on income alone, but still systematically under-serve women in financial hardship — the most vulnerable group. We computed all four gender × socioeconomic intersections. Standard marginal audits miss this. |
| ★ What is the `reason_text` column? | A plain-English sentence generated from each student's top-3 SHAP factors. E.g. "Grade fell 2.3pts; 67% unit non-completion; outstanding debt." Advisors read this directly — they do not need to understand SHAP. |
| What is the financial stress index? | A composite feature: debtor status (weighted 2×) + overdue tuition fees (2×) + absence of scholarship (1×). Range 0–5. Heavier weighting on debt and overdue fees reflects their stronger empirical correlation with dropout. |
| How does your pipeline handle data drift in production? | Honest answer: out of scope for this problem statement. In production we would add PSI monitoring on `grade_delta` and `financial_stress_index`, and trigger retraining when PSI exceeds 0.20. |

### 9.2 Presentation Structure (3–4 slides)

1. **Problem & approach** — one sentence on the problem, one on the solution, one on why fair and calibrated prediction matters more than raw accuracy.
2. **Pipeline architecture** — Bronze/Silver/Gold diagram, four table names, MLflow experiment name. Do not over-explain — judges know Databricks.
3. **Results** — XGBoost AUC, calibrated reliability diagram, intersectional fairness audit numbers, sample Gold table row showing a real at-risk student with `reason_text` and their top 3 SHAP factors.
4. **Fairness findings** — be specific and honest. Name the intersectional disparity, name the cause, name the production mitigation. This slide wins the 25%.

### 9.3 Notebook Hygiene Checklist

- Every notebook cell has a Markdown heading explaining what it does.
- MLflow run names are meaningful — not "Run 1" or "Untitled".
- All Delta table writes use `.mode("overwrite")` — pipeline is idempotent.
- Bronze table has zero modifications from the raw CSV.
- ★ Calibration section is a dedicated notebook cell with a Markdown heading.
- Fairness audit has a dedicated notebook section — marginal and intersectional in sub-cells.
- ★ `reason_text` generation is its own cell with examples printed for review.
- Gold table write is the last code cell — Gold is always the final output.
- A `README.md` file exists at the repo root explaining how to run the pipeline end-to-end.
- No hard-coded file paths that only work on one team member's cluster.

---

## 10. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Databricks cluster not provisioned | Medium | Pre-provision before hour 0. Fallback: local Jupyter with pandas + sklearn. |
| SHAP import fails on cluster | Low | `pip install shap` at notebook top. Fallback: `KernelExplainer` on 100-row sample. |
| `fairlearn` not on cluster | Medium | All fairness metrics are implementable in 20 lines of pandas — no library needed. |
| XGBoost AUC lower than Logistic Regression | Low | Document in MLflow and register whichever model has higher AUC. |
| Unity Catalog not enabled | Low | Write tables to default catalog. Note deviation in README. |
| ★ Calibration degrades AUC on val set | Low | `CalibratedClassifierCV` with `cv="prefit"` preserves the original fit — only the probability surface changes, not rankings. |
| ★ Intersectional group too small for stable metrics | Medium | If any intersection has < 30 positive examples, flag the metric as "n too small for reliable estimate" in the findings doc. Do not omit — document the limitation. |
| `reason_text` has null or generic output | Medium | Pre-test the `factor_interpretations` dict against actual Silver column names at hour 14. Ensure feature names match exactly. |
| Time overrun on model training | Medium | Hard cap model training at hour 9. Default hyperparameters are sufficient. |
| OULAD bonus attempted and breaks pipeline | High (if early) | Never attempt OULAD before hour 21. Run in a separate notebook. |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| Bronze layer | Raw, unmodified data ingested directly from source. Immutable. |
| Silver layer | Cleaned, validated, and feature-enriched version of Bronze. Primary training data source. |
| Gold layer | Business-ready aggregated output. At-risk student table with calibrated scores, `reason_text`, and interventions. |
| ★ Platt calibration | A post-processing step (`CalibratedClassifierCV`, `method="sigmoid"`) that maps raw model probabilities to true calibrated probabilities. Validated with a reliability diagram logged to MLflow. |
| ★ Intersectional fairness | Fairness computed across the cross-product of two protected attributes (e.g. gender × socioeconomic group). Detects disparities invisible to marginal audits. |
| ★ `reason_text` | A plain-English sentence built from each student's top-3 SHAP factors. Enables non-technical advisors to act without understanding the model. |
| Demographic parity difference | `\|P(pred=1\|group A) − P(pred=1\|group B)\|`. Ideal: 0. Threshold: >0.10 is significant. |
| Equal opportunity difference | `\|TPR(group A) − TPR(group B)\|`. Measures whether the model catches actual dropouts equally across groups. |
| SHAP | SHapley Additive exPlanations. Positive SHAP value = feature pushes toward dropout prediction. |
| MLflow Model Registry | Centralised model store tracking versions and lifecycle stages (Staging, Production, Archived). |
| Unity Catalog | Databricks governance layer for Delta tables — access control, lineage, audit logging. |
| `scale_pos_weight` | XGBoost parameter weighting the positive class. Set to (count negatives / count positives). |
| Intervention tier | Categorical urgency: high (≥0.70 calibrated), medium (0.40–0.70), low (<0.40). |
| PSI | Population Stability Index. Data drift metric. Not required here — noted as production extension. |
| `TreeExplainer` | SHAP explainer optimised for tree-based models. Faster and exact vs `KernelExplainer`. |

---

*The Dropout Signal — PRD v2.0 | April 2026 | Confidential — Hackathon Use Only | ★ = Differentiator added in v2.0*
