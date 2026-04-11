# Databricks notebook source

# MAGIC %md
# MAGIC # Notebook 03 — Model Training & MLflow
# MAGIC ## The Dropout Signal: Logistic Regression, XGBoost, ★ Platt Calibration
# MAGIC
# MAGIC **Purpose:** Train two models, log everything to MLflow, calibrate the champion, and register it.
# MAGIC
# MAGIC **Input Table:** `silver.uci_dropout_clean`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - MLflow experiment `dropout_signal_hackathon` with 2 runs
# MAGIC - Calibrated champion model registered as `dropout_risk_champion`
# MAGIC - Test predictions saved to `silver.model_test_results`
# MAGIC
# MAGIC **★ Differentiator:** Platt calibration makes risk scores statistically meaningful probabilities.

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

print("✅ All imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Load Silver Table & Convert to Pandas

# COMMAND ----------

df_spark = spark.table("silver.uci_dropout_clean")
df = df_spark.toPandas()

print(f"✅ Silver table loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   dropout_label distribution: {df['dropout_label'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Define Features & Split Data
# MAGIC
# MAGIC **Features:** All Silver columns except `Target` (string), `dropout_label` (target), and `student_id` (identifier).
# MAGIC
# MAGIC **Split:** 80% train / 20% test, stratified on `dropout_label`, random_state=42.
# MAGIC
# MAGIC **Validation:** 10% carved from training set for Platt calibration.

# COMMAND ----------

# Define feature columns — exclude target, label, and ID
EXCLUDE_COLS = ["Target", "dropout_label", "student_id"]
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

# Further split training: 90% train, 10% validation (for calibration)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
)

# Preserve student IDs for the test set
test_student_ids = df.loc[X_test.index, "student_id"].values

print(f"✅ Data split complete:")
print(f"   Train:      {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Validation: {X_val.shape[0]} rows (for Platt calibration)")
print(f"   Test:       {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"   Features:   {X_train.shape[1]}")

# Compute scale_pos_weight for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\n   Class balance (train): {neg_count} negative, {pos_count} positive")
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Set MLflow Experiment

# COMMAND ----------

EXPERIMENT_NAME = "/dropout_signal_hackathon"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✅ MLflow experiment set: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Model 1 — Logistic Regression Baseline
# MAGIC
# MAGIC | Parameter | Value |
# MAGIC |-----------|-------|
# MAGIC | C | 1.0 |
# MAGIC | max_iter | 1000 |
# MAGIC | class_weight | balanced |
# MAGIC | solver | lbfgs |

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression_baseline") as lr_run:
    
    # ----- Model Definition -----
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    
    # ----- Training -----
    lr_model.fit(X_train, y_train)
    
    # ----- Predictions -----
    lr_preds = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # ----- Metrics -----
    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "roc_auc": roc_auc_score(y_test, lr_proba),
        "f1_macro": f1_score(y_test, lr_preds, average="macro"),
        "precision": precision_score(y_test, lr_preds),
        "recall": recall_score(y_test, lr_preds),
    }
    
    # ----- Log Parameters -----
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X_train.shape[1],
        "random_state": 42,
    })
    
    # ----- Log Metrics -----
    mlflow.log_metrics(lr_metrics)
    
    # ----- Confusion Matrix -----
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, lr_preds, ax=ax, cmap="Blues")
    ax.set_title("Logistic Regression — Confusion Matrix")
    fig.tight_layout()
    fig.savefig("/tmp/lr_confusion_matrix.png", dpi=150)
    mlflow.log_artifact("/tmp/lr_confusion_matrix.png")
    plt.close()
    
    # ----- Log Model -----
    mlflow.sklearn.log_model(lr_model, "model")
    
    lr_run_id = lr_run.info.run_id
    print("✅ Logistic Regression baseline logged to MLflow")
    print(f"   Run ID: {lr_run_id}")
    for metric, value in lr_metrics.items():
        print(f"   {metric}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Model 2 — XGBoost Classifier
# MAGIC
# MAGIC | Parameter | Value |
# MAGIC |-----------|-------|
# MAGIC | n_estimators | 200 |
# MAGIC | max_depth | 5 |
# MAGIC | learning_rate | 0.1 |
# MAGIC | scale_pos_weight | auto-computed |
# MAGIC | eval_metric | auc |

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_classifier") as xgb_run:
    
    # ----- Model Definition -----
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False,
    )
    
    # ----- Training -----
    xgb_model.fit(X_train, y_train)
    
    # ----- Predictions (raw — before calibration) -----
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # ----- Metrics -----
    xgb_metrics = {
        "accuracy": accuracy_score(y_test, xgb_preds),
        "roc_auc": roc_auc_score(y_test, xgb_proba),
        "f1_macro": f1_score(y_test, xgb_preds, average="macro"),
        "precision": precision_score(y_test, xgb_preds),
        "recall": recall_score(y_test, xgb_preds),
    }
    
    # ----- Log Parameters -----
    mlflow.log_params({
        "model_type": "XGBClassifier",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "eval_metric": "auc",
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X_train.shape[1],
        "random_state": 42,
    })
    
    # ----- Log Metrics -----
    mlflow.log_metrics(xgb_metrics)
    
    # ----- Confusion Matrix -----
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, xgb_preds, ax=ax, cmap="Oranges")
    ax.set_title("XGBoost — Confusion Matrix")
    fig.tight_layout()
    fig.savefig("/tmp/xgb_confusion_matrix.png", dpi=150)
    mlflow.log_artifact("/tmp/xgb_confusion_matrix.png")
    plt.close()
    
    # ----- Feature Importance -----
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")
    
    # ----- Log Raw XGBoost Model (for SHAP in Notebook 05) -----
    mlflow.sklearn.log_model(xgb_model, "raw_xgb_model")
    
    # ----- Also log the main model -----
    mlflow.sklearn.log_model(xgb_model, "model")
    
    xgb_run_id = xgb_run.info.run_id
    print("✅ XGBoost classifier logged to MLflow")
    print(f"   Run ID: {xgb_run_id}")
    for metric, value in xgb_metrics.items():
        print(f"   {metric}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 Model Comparison

# COMMAND ----------

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'Logistic Regression':>20} {'XGBoost':>20}")
print("-" * 60)
for metric in ["accuracy", "roc_auc", "f1_macro", "precision", "recall"]:
    lr_val = lr_metrics[metric]
    xgb_val = xgb_metrics[metric]
    winner = " ◀" if xgb_val >= lr_val else ""
    print(f"{metric:<20} {lr_val:>20.4f} {xgb_val:>20.4f}{winner}")

champion = "XGBoost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "Logistic Regression"
print(f"\n🏆 Champion model (by ROC AUC): {champion}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8 ★ Platt Calibration
# MAGIC
# MAGIC **Why:** Raw XGBoost `predict_proba` output is NOT a calibrated probability.
# MAGIC A score of 0.7 does not mean a 70% dropout chance.
# MAGIC
# MAGIC **Method:** `CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")`
# MAGIC fitted on the held-out validation split.
# MAGIC
# MAGIC **After calibration:** the intervention tier thresholds (0.40, 0.70) carry statistical meaning.

# COMMAND ----------

# ★ Apply Platt scaling on the validation set
calibrated_model = CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")
calibrated_model.fit(X_val, y_val)

# Get calibrated predictions on the test set
calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
calibrated_preds = calibrated_model.predict(X_test)

# Calibrated metrics
cal_metrics = {
    "accuracy": accuracy_score(y_test, calibrated_preds),
    "roc_auc": roc_auc_score(y_test, calibrated_proba),
    "f1_macro": f1_score(y_test, calibrated_preds, average="macro"),
    "precision": precision_score(y_test, calibrated_preds),
    "recall": recall_score(y_test, calibrated_preds),
}

print("★ Calibrated XGBoost metrics:")
for metric, value in cal_metrics.items():
    diff = value - xgb_metrics[metric]
    direction = "↑" if diff >= 0 else "↓"
    print(f"   {metric}: {value:.4f} ({direction}{abs(diff):.4f} vs raw)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ★ Reliability Diagram
# MAGIC
# MAGIC Compares predicted probabilities vs actual fraction of positives.
# MAGIC A perfectly calibrated model follows the diagonal.

# COMMAND ----------

# Generate reliability diagram — comparing raw vs calibrated
fig, ax = plt.subplots(figsize=(7, 6))

# Raw XGBoost
prob_true_raw, prob_pred_raw = calibration_curve(y_test, xgb_proba, n_bins=10)
ax.plot(prob_pred_raw, prob_true_raw, marker="s", label="XGBoost (raw)", color="#e74c3c", linewidth=2)

# Calibrated XGBoost
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_proba, n_bins=10)
ax.plot(prob_pred_cal, prob_true_cal, marker="o", label="XGBoost (calibrated ★)", color="#2ecc71", linewidth=2)

# Perfect calibration line
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration", linewidth=1)

ax.set_xlabel("Mean predicted probability", fontsize=12)
ax.set_ylabel("Fraction of positives", fontsize=12)
ax.set_title("★ Reliability Diagram — Platt Calibration", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("/tmp/reliability_diagram.png", dpi=150)
plt.show()

print("✅ Reliability diagram saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Calibration Artifacts to MLflow

# COMMAND ----------

# Log calibration artifacts to the XGBoost run
with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact("/tmp/reliability_diagram.png")
    mlflow.log_params({
        "calibration_method": "platt_scaling",
        "calibration_val_size": X_val.shape[0],
    })
    # Log calibrated metrics with prefix
    for metric, value in cal_metrics.items():
        mlflow.log_metric(f"calibrated_{metric}", value)
    
    # Log the calibrated model
    mlflow.sklearn.log_model(calibrated_model, "calibrated_model")

print("✅ Calibration artifacts logged to MLflow run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.9 Register Champion Model
# MAGIC
# MAGIC Register the **calibrated** model (not the raw XGBoost) as `dropout_risk_champion`.

# COMMAND ----------

MODEL_NAME = "dropout_risk_champion"

# Register the calibrated model
model_uri = f"runs:/{xgb_run_id}/calibrated_model"
registered_model = mlflow.register_model(model_uri, MODEL_NAME)

print(f"✅ Calibrated model registered as: {MODEL_NAME}")
print(f"   Version: {registered_model.version}")

# COMMAND ----------

# Set tags on the registered model
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.set_registered_model_tag(MODEL_NAME, "dataset", "uci_dropout")
client.set_registered_model_tag(MODEL_NAME, "version", "2.0")
client.set_registered_model_tag(MODEL_NAME, "fairness_audited", "pending")
client.set_registered_model_tag(MODEL_NAME, "calibrated", "true")

# Transition to Production stage
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered_model.version,
    stage="Production"
)

print(f"✅ Model transitioned to Production stage")
print(f"   Tags: dataset=uci_dropout, version=2.0, calibrated=true, fairness_audited=pending")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.10 Save Test Results for Downstream Notebooks
# MAGIC
# MAGIC Save the complete test set with predictions to `silver.model_test_results`.
# MAGIC This table is used by notebooks 04 (Fairness), 05 (SHAP), and 06 (Gold).

# COMMAND ----------

# Build test results DataFrame
test_results = X_test.copy()
test_results["student_id"] = test_student_ids
test_results["dropout_label"] = y_test.values
test_results["dropout_pred"] = calibrated_preds
test_results["risk_score"] = calibrated_proba

# Also store the original Target value
test_results["Target"] = df.loc[X_test.index, "Target"].values

# Convert to Spark DataFrame and save
test_results_spark = spark.createDataFrame(test_results)
test_results_spark.write.format("delta").mode("overwrite").saveAsTable("silver.model_test_results")

print(f"✅ Test results saved to silver.model_test_results")
print(f"   Rows: {test_results_spark.count()}")
print(f"   Columns: {len(test_results_spark.columns)}")
print(f"   Includes: all features + student_id + dropout_label + dropout_pred + risk_score")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.11 Save Metadata for Downstream Notebooks
# MAGIC
# MAGIC Save the XGBoost run ID and feature columns list for use in notebooks 04, 05, 06.

# COMMAND ----------

import json

metadata = {
    "xgb_run_id": xgb_run_id,
    "lr_run_id": lr_run_id,
    "feature_columns": FEATURE_COLS,
    "champion_model_name": MODEL_NAME,
    "experiment_name": EXPERIMENT_NAME,
}

# Save as JSON artifact in the XGBoost run
with open("/tmp/pipeline_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact("/tmp/pipeline_metadata.json")

print("✅ Pipeline metadata saved to MLflow")
print(f"   XGBoost Run ID: {xgb_run_id}")
print(f"   LR Run ID: {lr_run_id}")
print(f"   Feature columns: {len(FEATURE_COLS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Model Training Complete
# MAGIC
# MAGIC **Deliverables:**
# MAGIC - ✅ MLflow experiment `dropout_signal_hackathon` with 2 runs
# MAGIC - ✅ Logistic Regression baseline: all params/metrics logged
# MAGIC - ✅ XGBoost classifier: all params/metrics logged
# MAGIC - ✅ ★ Platt calibration applied — `reliability_diagram.png` logged
# MAGIC - ✅ ★ Calibrated model registered as `dropout_risk_champion` at Production stage
# MAGIC - ✅ Test predictions saved to `silver.model_test_results`
# MAGIC
# MAGIC **Next:** Run `04_fairness_audit` to compute marginal and intersectional fairness metrics.
