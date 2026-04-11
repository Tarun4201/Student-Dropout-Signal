# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 03 — Model Training & MLflow
# MAGIC ## The Dropout Signal: Logistic Regression, XGBoost, ★ Platt Calibration
# MAGIC
# MAGIC **Input Table:** `silver.uci_dropout_clean`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - MLflow experiment `dropout_signal_hackathon` with 2 runs
# MAGIC - Calibrated champion model registered as `dropout_risk_champion`
# MAGIC - Test predictions saved to `silver.model_test_results`

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Imports

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from mlflow.models import infer_signature

print("✅ All imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Load Silver Table

# COMMAND ----------

df_spark = spark.table("silver.uci_dropout_clean")
df = df_spark.toPandas()

print(f"✅ Silver table loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   dropout_label distribution: {df['dropout_label'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Define Features & Split Data

# COMMAND ----------

# Exclude target, label, and ID columns
EXCLUDE_COLS = ["target", "dropout_label", "student_id"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

X = df[FEATURE_COLS]
y = df["dropout_label"]

print(f"Feature columns ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {col}")

# COMMAND ----------

# 80/20 stratified split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 90/10 validation from training (for calibration)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
)

test_student_ids = df.loc[X_test.index, "student_id"].values

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"✅ Data split:")
print(f"   Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 MLflow Experiment

# COMMAND ----------

EXPERIMENT_NAME = "/dropout_signal_hackathon"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✅ MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Model 1 — Logistic Regression Baseline

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression_baseline") as lr_run:
    lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "roc_auc": roc_auc_score(y_test, lr_proba),
        "f1_macro": f1_score(y_test, lr_preds, average="macro"),
        "precision": precision_score(y_test, lr_preds),
        "recall": recall_score(y_test, lr_preds),
    }
    
    mlflow.log_params({"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000,
                        "class_weight": "balanced", "solver": "lbfgs",
                        "train_size": X_train.shape[0], "val_size": X_val.shape[0],
                        "test_size": X_test.shape[0], "n_features": X_train.shape[1], "random_state": 42})
    mlflow.log_metrics(lr_metrics)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, lr_preds, ax=ax, cmap="Blues")
    ax.set_title("Logistic Regression — Confusion Matrix")
    fig.tight_layout(); fig.savefig("/tmp/lr_confusion_matrix.png", dpi=150)
    mlflow.log_artifact("/tmp/lr_confusion_matrix.png"); plt.close()
    
    lr_signature = infer_signature(X_train, lr_model.predict(X_train))
    mlflow.sklearn.log_model(lr_model, "model", signature=lr_signature, input_example=X_train[:5])
    lr_run_id = lr_run.info.run_id
    
    print("✅ Logistic Regression logged")
    for m, v in lr_metrics.items(): print(f"   {m}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Model 2 — XGBoost Classifier

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_classifier") as xgb_run:
    xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                               scale_pos_weight=scale_pos_weight, eval_metric="auc",
                               random_state=42, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_metrics = {
        "accuracy": accuracy_score(y_test, xgb_preds),
        "roc_auc": roc_auc_score(y_test, xgb_proba),
        "f1_macro": f1_score(y_test, xgb_preds, average="macro"),
        "precision": precision_score(y_test, xgb_preds),
        "recall": recall_score(y_test, xgb_preds),
    }
    
    mlflow.log_params({"model_type": "XGBClassifier", "n_estimators": 200, "max_depth": 5,
                        "learning_rate": 0.1, "scale_pos_weight": round(scale_pos_weight, 4),
                        "eval_metric": "auc", "train_size": X_train.shape[0],
                        "val_size": X_val.shape[0], "test_size": X_test.shape[0],
                        "n_features": X_train.shape[1], "random_state": 42})
    mlflow.log_metrics(xgb_metrics)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, xgb_preds, ax=ax, cmap="Oranges")
    ax.set_title("XGBoost — Confusion Matrix")
    fig.tight_layout(); fig.savefig("/tmp/xgb_confusion_matrix.png", dpi=150)
    mlflow.log_artifact("/tmp/xgb_confusion_matrix.png"); plt.close()
    
    importance_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": xgb_model.feature_importances_}).sort_values("importance", ascending=False)
    importance_df.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")
    
    xgb_signature = infer_signature(X_train, xgb_model.predict(X_train))
    mlflow.sklearn.log_model(xgb_model, "raw_xgb_model", signature=xgb_signature, input_example=X_train[:5])
    mlflow.sklearn.log_model(xgb_model, "model", signature=xgb_signature, input_example=X_train[:5])
    
    xgb_run_id = xgb_run.info.run_id
    print("✅ XGBoost logged")
    for m, v in xgb_metrics.items(): print(f"   {m}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 Model Comparison

# COMMAND ----------

print(f"{'Metric':<20} {'LogReg':>12} {'XGBoost':>12}")
print("-" * 46)
for m in ["accuracy", "roc_auc", "f1_macro", "precision", "recall"]:
    print(f"{m:<20} {lr_metrics[m]:>12.4f} {xgb_metrics[m]:>12.4f}")
champion = "XGBoost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "LogReg"
print(f"\n🏆 Champion: {champion}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8 ★ Platt Calibration

# COMMAND ----------

calibrated_model = CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")
calibrated_model.fit(X_val, y_val)

calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
calibrated_preds = calibrated_model.predict(X_test)

cal_metrics = {
    "accuracy": accuracy_score(y_test, calibrated_preds),
    "roc_auc": roc_auc_score(y_test, calibrated_proba),
    "f1_macro": f1_score(y_test, calibrated_preds, average="macro"),
    "precision": precision_score(y_test, calibrated_preds),
    "recall": recall_score(y_test, calibrated_preds),
}

print("★ Calibrated metrics:")
for m, v in cal_metrics.items():
    diff = v - xgb_metrics[m]
    print(f"   {m}: {v:.4f} ({'↑' if diff >= 0 else '↓'}{abs(diff):.4f})")

# COMMAND ----------

# ★ Reliability diagram
fig, ax = plt.subplots(figsize=(7, 6))
prob_true_raw, prob_pred_raw = calibration_curve(y_test, xgb_proba, n_bins=10)
ax.plot(prob_pred_raw, prob_true_raw, marker="s", label="XGBoost (raw)", color="#e74c3c", linewidth=2)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_proba, n_bins=10)
ax.plot(prob_pred_cal, prob_true_cal, marker="o", label="XGBoost (calibrated ★)", color="#2ecc71", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("★ Reliability Diagram"); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("/tmp/reliability_diagram.png", dpi=150); plt.show()

# COMMAND ----------

# Log calibration to MLflow
with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact("/tmp/reliability_diagram.png")
    mlflow.log_params({"calibration_method": "platt_scaling", "calibration_val_size": X_val.shape[0]})
    for m, v in cal_metrics.items(): mlflow.log_metric(f"calibrated_{m}", v)
    calibrated_signature = infer_signature(X_val, calibrated_model.predict(X_val))
    mlflow.sklearn.log_model(calibrated_model, "calibrated_model", signature=calibrated_signature, input_example=X_val[:5])
print("✅ Calibration logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.9 Register Champion Model

# COMMAND ----------

MODEL_NAME = "dropout_risk_champion"
model_uri = f"runs:/{xgb_run_id}/calibrated_model"
registered_model = mlflow.register_model(model_uri, MODEL_NAME)

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_tag(MODEL_NAME, "dataset", "uci_dropout")
client.set_registered_model_tag(MODEL_NAME, "version", "2.0")
client.set_registered_model_tag(MODEL_NAME, "fairness_audited", "pending")
client.set_registered_model_tag(MODEL_NAME, "calibrated", "true")
client.set_registered_model_alias(MODEL_NAME, "production", registered_model.version)

print(f"✅ Model registered: {MODEL_NAME} v{registered_model.version} → production alias")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.10 Save Test Results

# COMMAND ----------

test_results = X_test.copy()
test_results["student_id"] = test_student_ids
test_results["dropout_label"] = y_test.values
test_results["dropout_pred"] = calibrated_preds
test_results["risk_score"] = calibrated_proba
test_results["target"] = df.loc[X_test.index, "target"].values

test_results_spark = spark.createDataFrame(test_results)
test_results_spark.write.format("delta").mode("overwrite").saveAsTable("silver.model_test_results")

print(f"✅ Test results saved: silver.model_test_results ({test_results_spark.count()} rows)")

# COMMAND ----------

# Save metadata
import json
metadata = {"xgb_run_id": xgb_run_id, "lr_run_id": lr_run_id, "feature_columns": FEATURE_COLS,
            "champion_model_name": MODEL_NAME, "experiment_name": EXPERIMENT_NAME}
with open("/tmp/pipeline_metadata.json", "w") as f: json.dump(metadata, f, indent=2)
with mlflow.start_run(run_id=xgb_run_id): mlflow.log_artifact("/tmp/pipeline_metadata.json")
print(f"✅ Metadata saved | XGB Run ID: {xgb_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Model Training Complete
# MAGIC **Next:** Run `04_fairness_audit`
