# 🎓 The Dropout Signal
**Fair Early-Warning Pipeline for Student Dropout Prediction**

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=for-the-badge)

An end-to-end, fair, and explainable machine learning pipeline built on Databricks to predict student dropout risk. This pipeline transforms raw student data into actionable risk assessments for academic advisors.

> **Winner Feature:** Built for non-technical advisors via natural language generation of `reason_text` and robust Platt Calibration for statistically sound intervention tiers.

---

## 🏗️ Architecture (Medallion Lakehouse)

The pipeline is implemented natively in Databricks using a 6-notebook structure mapping to the Medallion Architecture:

1. **Bronze (`01_bronze_layer`)**: Idempotent ingest of raw data using Unity Catalog.
2. **Silver (`02_silver_layer`)**: Feature engineering (`grade_delta`, `financial_stress_index`, `engagement_score`).
3. **Model Training (`03_model_training`)**: XGBoost classification vs Logistic Regression baseline, complete with **Platt scaling** to calibrate output probabilities and logged via **MLflow**.
4. **Fairness Audit (`04_fairness_audit`)**: Comprehensive **intersectional fairness audit** to establish demographic parity across vulnerable demographic variables. 
5. **SHAP Explainability (`05_shap_explainability`)**: TreeExplainer calculation to identify the top 3 global and localized factors driving attrition risk.
6. **Gold (`06_gold_table`)**: Curated table featuring human-readable `reason_text` strings generating plain-English intervention insights directly from SHAP values.

---

## 🌟 Key Differentiators

1. **Jargon-Free Explainability (`reason_text`)**: We abstract away complex SHAP value arrays into clean, advisor-ready sentences (e.g. *"Grade fell 2.3pts; 67% non-completion; debt outstanding."*). 
2. **Statistically Defensible Interventions (Platt Calibration)**: Rather than arbitrary risk thresholds, our model applies Platt Scaling to calibrate probability scores. A `0.85` score is true 85% risk, allowing for accurate mapping to High/Medium/Low priority tiers.
3. **Intersectional Fairness**: Marginal fairness checks easily conceal systemic bias against highly vulnerable subgroups (e.g. low-income female students). We conduct granular intersectional parity tests.

---

## 🚀 Getting Started

### Prerequisites
- A Databricks workspace
- ML Runtime (version 13.3 LTS ML or higher recommended)
- `xgboost` and `shap` python packages installed on the cluster.

### Execution
Upload the 6 notebooks from the `/notebooks` sequence to a Databricks Workspace folder and execute them chronologically:
1. `01_bronze_layer.py`
2. `02_silver_layer.py`
3. `03_model_training.py`
4. `04_fairness_audit.py`
5. `05_shap_explainability.py`
6. `06_gold_table.py`

### Final Deliverable
The final actionable dataset is produced in the Unity Catalog table:
```sql
SELECT student_id, risk_score, intervention_tier, reason_text 
FROM gold.at_risk_students
WHERE intervention_tier IN ('HIGH', 'MEDIUM')
ORDER BY risk_score DESC;
```

---
*Built for a 24-hour hackathon.*
