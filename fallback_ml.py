import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import shap

_model = None
_explainer = None
_feature_cols = None

def train_fallback_model(df, target_col='dropout_label', exclude_cols=None):
    """
    Trains a local ML model on the provided dataframe.
    This acts as our fallback when Databricks is unavailable.
    """
    global _model, _explainer, _feature_cols
    
    if exclude_cols is None:
        exclude_cols = ['target', 'dropout_label', 'student_id', 'reason_text', 
                        'intervention_tier', 'socioeconomic_group', 'gender_label', 
                        'intersection', 'risk_score', 'dropout_predicted', 'top_shap_factors',
                        'sentiment_score', 'red_zone']
        
    _feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[_feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"🌲 Training local fallback HistGradientBoosting model on {X.shape[0]} rows and {X.shape[1]} features...")
    
    # We use HistGradientBoostingClassifier since XGBoost fails on Mac without libomp
    _model = HistGradientBoostingClassifier(
        max_iter=100, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=42
    )
    _model.fit(X, y)
    
    print("🌲 Model trained successfully. Initializing SHAP explainer...")
    
    try:
        # HistGradientBoosting works with TreeExplainer, but fallback to Permutation if needed
        _explainer = shap.TreeExplainer(_model)
    except Exception as e:
        print(f"⚠️ TreeExplainer failed ({e}). Falling back to general Explainer.")
        # Generate a background dataset for the general explainer (100 samples)
        background = shap.sample(X, 100)
        _explainer = shap.Explainer(_model.predict_proba, background)
        
    print("✅ Fallback ML pipeline ready.")

def get_risk_scores(df):
    """
    Generates probability scores using the trained local XGBoost model.
    """
    if _model is None:
        raise ValueError("Model has not been trained yet. Call train_fallback_model first.")
    
    # Ensure columns match, handle missing columns by filling with 0
    X_pred = df[df.columns.intersection(_feature_cols)].copy()
    for col in _feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[_feature_cols]

    # Return probabilities for class 1 (Dropout)
    return _model.predict_proba(X_pred)[:, 1]

def get_shap_factors(df):
    """
    Generates top 3 SHAP factors for each row in the dataframe.
    Returns a list of JSON strings representing the factors.
    """
    if _explainer is None:
        raise ValueError("Explainer has not been initialized. Call train_fallback_model first.")
    
    X_pred = df[df.columns.intersection(_feature_cols)].copy()
    for col in _feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[_feature_cols]

    # Calculate SHAP values
    shap_values = _explainer.shap_values(X_pred)
    
    results = []
    # Depending on xgb/shap versions, shap_values might be a list (multiclass) or array
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] # Take positive class
    else:
        shap_vals = shap_values
    
    for i in range(len(shap_vals)):
        row_shap = shap_vals[i]
        
        # We only care about positive push factors (what pushes risk up)
        # Pair feature names with their shap values
        push_factors = [(f, v) for f, v in zip(_feature_cols, row_shap) if v > 0]
        
        # Sort descending by impact
        push_factors.sort(key=lambda x: x[1], reverse=True)
        
        top3 = push_factors[:3]
        
        # Clean names
        factors_json = []
        for name, _ in top3:
            clean_name = name.replace('_', ' ').title().replace('Sem ', 'Semester ')
            factors_json.append(clean_name)
        
        # Add fallback if no positive push factors
        if not factors_json:
            factors_json = ["General Poor Performance", "Undisclosed At-Risk Factor"]
            
        results.append(json.dumps(factors_json))
        
    return results

def is_ready():
    return _model is not None and _explainer is not None
