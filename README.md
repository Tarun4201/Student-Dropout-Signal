# 🎓 The Dropout Signal

Welcome to **The Dropout Signal**! This is a fair early-warning machine learning system built to predict student dropout risk. It gives academic advisors clear, actionable insights to help struggling students.

🌍 **Live Demo:** [View the Project Website](https://47fc5a2984d6ed08-103-215-237-11.serveousercontent.com) *(Note: As this is a free tunnel, you might see a brief Serveo warning page before the site loads. Please click to continue).*

---

## 🤖 Our ML Pipeline Explained

Our Machine Learning pipeline is designed to be highly accurate, explainable, and fair. Instead of just giving a black-box percentage, we break down *why* a student is at risk so that advisors know exactly how to help.

Here is how our ML Pipeline works:

1. **Feature Engineering:** We take basic student data and create key behavioral indicators, such as a student's `grade_delta` (how much their grades are dropping) and their `financial_stress_index`.
2. **Model Training:** We train an **XGBoost Classifier** (a powerful decision-tree-based model) on historical student data to predict the likelihood of a student dropping out.
3. **Platt Calibration:** Machine learning models often produce raw numbers that don't translate well to real-world probabilities. We apply a statistical method called *Platt Scaling* so a risk score of `0.85` actually means an 85% probability of dropping out.
4. **Fairness Auditing:** We run intersectional fairness audits to ensure our model doesn't unintentionally target or neglect vulnerable groups (e.g., low-income female students). We want equitable predictions for everyone.
5. **Human-Readable Explanations (SHAP):** We use an AI explainability tool called **SHAP** to see exactly which factors push a student's risk up. We translate these raw math values into a plain-English `reason_text` (e.g., *"Grade fell 2.3pts; financial stress is high"*).

### Local "Fallback" Pipeline
Our main pipeline is built on Databricks. However, if the cloud is unreachable, we automatically shift to a local **Fallback ML Pipeline**. This uses a lightweight `HistGradientBoostingClassifier` directly within our web app to continue generating accurate predictions and explainable SHAP factors without any downtime!

---

## 🚀 Running Locally

If you'd like to run this on your own machine:

```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Start the web dashboard
python app.py
```
Then simply open `http://localhost:5050` in your web browser!
