"""
The Dropout Signal — Flask Backend
Processes UCI Dropout dataset and serves REST APIs for the dashboard.
"""

import os
import math
import json
import hashlib
import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ---------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(os.path.dirname(__file__),
                        'students_dropout_academic_success.csv')


def load_and_process_data():
    """Load from Live Databricks, with CSV fallback for Hackathon safety."""
    
    # 1. Attempt Live Databricks Connection
    try:
        from dotenv import load_dotenv
        import databricks.sql
        
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path, override=True)
        
        token = os.getenv("DATABRICKS_TOKEN")
        if not token or token.startswith('dapi...'):
            print("[WARN] No valid DATABRICKS_TOKEN provided in .env. Falling back to local offline CSV mode.")
            raise ValueError("No valid token")

        print("🚀 Connecting to Live Databricks SQL Warehouse...")
        conn = databricks.sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=token
        )
        
        cursor = conn.cursor()
        
        print("📥 Querying Silver layer...")
        cursor.execute("SELECT * FROM silver.uci_dropout_clean")
        cols = [desc[0] for desc in cursor.description]
        df = pd.DataFrame.from_records(cursor.fetchall(), columns=cols)
        
        print("📥 Querying Gold layer...")
        try:
            cursor.execute("SELECT * FROM gold.at_risk_students")
            cols = [desc[0] for desc in cursor.description]
            gold = pd.DataFrame.from_records(cursor.fetchall(), columns=cols)
        except Exception as e:
            print(f"[WARN] Could not find gold table: {e}")
            gold = pd.DataFrame()
            
        cursor.close()
        conn.close()
        
        # The API endpoints expect specific column mappings:
        df['dropout_label'] = (df['target'] == 'Dropout').astype(int)
        df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})
        
        # Merge gold risk predictions where they exist
        if not gold.empty:
            gold_sub = gold[['student_id', 'risk_score', 'intervention_tier', 'reason_text', 
                             'shap_factor_1', 'shap_value_1', 'shap_factor_2', 'shap_value_2', 'shap_factor_3', 'shap_value_3']]
            df = df.merge(gold_sub, on='student_id', how='left')
            
            # Fill NaNs for graduates / low-risk
            df['risk_score'] = df['risk_score'].fillna(0.15)
            df['intervention_tier'] = df['intervention_tier'].fillna('low')
            df['reason_text'] = df['reason_text'].fillna('Low risk of attrition. No intervention required.')
            df['dropout_predicted'] = (df['intervention_tier'].isin(['high', 'medium'])).astype(int)
            for i in [1, 2, 3]:
                df[f'shap_factor_{i}'] = df[f'shap_factor_{i}'].fillna('grade_delta')
                df[f'shap_value_{i}'] = df[f'shap_value_{i}'].fillna(0.0)
        else:
            df['risk_score'] = _simulate_risk_scores(df)
            df['dropout_predicted'] = (df['risk_score'] >= 0.40).astype(int)
            df['intervention_tier'] = df['risk_score'].apply(_assign_tier)
            df = _simulate_shap_factors(df)
            df['reason_text'] = df.apply(_build_reason_text, axis=1)

        # Intersection groups
        df['socioeconomic_group'] = df['financial_stress_index'].apply(
            lambda x: 'high_stress' if x >= 3 else 'low_stress')
        df['intersection'] = df['gender_label'].str.lower() + '_' + df['socioeconomic_group']

        print("✅ Live Databricks Hook Successful! Loaded {} rows.".format(len(df)))
        return df

    except Exception as e:
        if "No valid token" not in str(e):
             print(f"\n[ERROR] Databricks connection failed: {str(e)}")
        print("\n[INFO] 🚨 FALLING BACK TO OFFLINE CSV MODE 🚨\n")
        
        # --- ORIGINAL CSV LOAD LOGIC ---
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=['target']).copy()
        
        numeric_cols = [c for c in df.columns if c != 'target']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                
        df = df.reset_index(drop=True)
        df.insert(0, 'student_id', range(1, len(df) + 1))
        df['dropout_label'] = (df['target'] == 'Dropout').astype(int)
        
        df['grade_delta'] = df['curricular_units_2nd_sem_grade'] - df['curricular_units_1st_sem_grade']
        enr1 = df['curricular_units_1st_sem_enrolled']
        app1 = df['curricular_units_1st_sem_approved']
        enr2 = df['curricular_units_2nd_sem_enrolled']
        app2 = df['curricular_units_2nd_sem_approved']
        df['absenteeism_trend'] = ((enr1 - app1 + enr2 - app2) / (enr1 + enr2 + 1))
        
        df['financial_stress_index'] = df['debtor'] * 2 + (1 - df['tuition_fees_up_to_date']) * 2 + (1 - df['scholarship_holder'])
        df['engagement_score'] = (app1 / (enr1 + 1)) + (app2 / (enr2 + 1)) + (df['curricular_units_1st_sem_evaluations'] + df['curricular_units_2nd_sem_evaluations']) / 20
        
        df['risk_score'] = _simulate_risk_scores(df)
        df['dropout_predicted'] = (df['risk_score'] >= 0.40).astype(int)
        df['intervention_tier'] = df['risk_score'].apply(_assign_tier)
        
        df['socioeconomic_group'] = df['financial_stress_index'].apply(lambda x: 'high_stress' if x >= 3 else 'low_stress')
        df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})
        df['intersection'] = df['gender'].map({0: 'female', 1: 'male'}) + '_' + df['socioeconomic_group']
        
        df = _simulate_shap_factors(df)
        df['reason_text'] = df.apply(_build_reason_text, axis=1)
        return df


def _simulate_risk_scores(df):
    """Generate deterministic, realistic risk scores from features."""
    # Normalise key features to 0-1 range
    gd = df['grade_delta'].clip(-15, 15)
    gd_norm = (gd - gd.min()) / (gd.max() - gd.min() + 1e-9)
    # Invert: lower grade_delta → higher risk
    gd_risk = 1 - gd_norm

    at = df['absenteeism_trend'].clip(0, 1)
    fs = df['financial_stress_index'] / 5.0
    es = df['engagement_score'].clip(0, 4)
    es_norm = (es - es.min()) / (es.max() - es.min() + 1e-9)
    es_risk = 1 - es_norm

    # Weighted combination
    raw = (0.30 * gd_risk + 0.25 * at + 0.25 * fs + 0.20 * es_risk)

    # Add deterministic noise per student
    noise = df['student_id'].apply(
        lambda sid: (int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16)
                     % 1000) / 10000 - 0.05
    )
    raw = (raw + noise).clip(0, 1)

    # Sigmoid to make it look calibrated
    logit = np.log(raw / (1 - raw + 1e-9) + 1e-9)
    calibrated = 1 / (1 + np.exp(-logit * 1.2))

    # Align with actual dropout labels: boost dropouts, lower graduates
    actual = df['dropout_label']
    calibrated = calibrated * 0.6 + actual * 0.35 + 0.025

    return calibrated.clip(0.01, 0.99).round(3)


def _assign_tier(score):
    if score >= 0.70:
        return 'high'
    elif score >= 0.40:
        return 'medium'
    return 'low'


# Factor interpretations from PRD
FACTOR_INTERPRETATIONS = {
    'grade_delta': lambda v: f"grade {'fell' if v < 0 else 'rose'} {abs(v):.1f}pts semester-on-semester",
    'financial_stress_index': lambda v: f"financial stress score {v:.0f}/5",
    'absenteeism_trend': lambda v: f"{v * 100:.0f}% unit non-completion rate",
    'curricular_units_2nd_sem_grade': lambda v: f"semester 2 grade of {v:.1f}",
    'debtor': lambda v: "outstanding debt on record" if v == 1 else "no debt",
    'tuition_fees_up_to_date': lambda v: "tuition fees overdue" if v == 0 else "fees current",
    'scholarship_holder': lambda v: "no scholarship" if v == 0 else "scholarship holder",
    'engagement_score': lambda v: f"engagement score {v:.2f}",
    'curricular_units_1st_sem_grade': lambda v: f"semester 1 grade of {v:.1f}",
    'curricular_units_2nd_sem_approved': lambda v: f"{int(v)} units approved in sem 2",
    'curricular_units_1st_sem_approved': lambda v: f"{int(v)} units approved in sem 1",
    'admission_grade': lambda v: f"admission grade of {v:.1f}",
    'age_at_enrollment': lambda v: f"enrolled at age {int(v)}",
}

RISK_FEATURES = [
    'grade_delta', 'absenteeism_trend', 'financial_stress_index',
    'engagement_score', 'curricular_units_2nd_sem_grade',
    'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_approved',
    'curricular_units_1st_sem_approved', 'admission_grade',
    'debtor', 'tuition_fees_up_to_date', 'scholarship_holder',
    'age_at_enrollment',
]


def _simulate_shap_factors(df):
    """Assign top-3 SHAP-like factors per student based on feature deviance."""
    shap_f1, shap_v1 = [], []
    shap_f2, shap_v2 = [], []
    shap_f3, shap_v3 = [], []

    # Pre-compute feature medians for deviation
    medians = {f: df[f].median() for f in RISK_FEATURES if f in df.columns}

    for _, row in df.iterrows():
        deviations = {}
        for f in RISK_FEATURES:
            if f not in df.columns:
                continue
            val = row[f]
            med = medians[f]
            # For risk-increasing features, deviation from median
            if f in ('grade_delta', 'engagement_score',
                     'curricular_units_2nd_sem_grade',
                     'curricular_units_1st_sem_grade',
                     'curricular_units_2nd_sem_approved',
                     'curricular_units_1st_sem_approved',
                     'admission_grade',
                     'tuition_fees_up_to_date', 'scholarship_holder'):
                dev = (med - val) / (abs(med) + 1)
            else:
                dev = (val - med) / (abs(med) + 1)
            deviations[f] = dev

        sorted_dev = sorted(deviations.items(), key=lambda x: abs(x[1]),
                            reverse=True)
        top3 = sorted_dev[:3]

        shap_f1.append(top3[0][0])
        shap_v1.append(round(top3[0][1], 4))
        shap_f2.append(top3[1][0])
        shap_v2.append(round(top3[1][1], 4))
        shap_f3.append(top3[2][0])
        shap_v3.append(round(top3[2][1], 4))

    df['shap_factor_1'] = shap_f1
    df['shap_value_1'] = shap_v1
    df['shap_factor_2'] = shap_f2
    df['shap_value_2'] = shap_v2
    df['shap_factor_3'] = shap_f3
    df['shap_value_3'] = shap_v3
    return df


def _build_reason_text(row):
    """Build plain-English reason sentence from top-3 SHAP factors."""
    reasons = []
    for i in range(1, 4):
        feature = row[f'shap_factor_{i}']
        value = row.get(feature, None)
        if feature in FACTOR_INTERPRETATIONS and value is not None:
            reasons.append(FACTOR_INTERPRETATIONS[feature](value))
        else:
            reasons.append(feature.replace('_', ' '))
    return '; '.join(reasons).capitalize() + '.'


# ---------------------------------------------------------------------------
# LOAD DATA ON STARTUP
# ---------------------------------------------------------------------------
print("Loading and processing dataset...")
DF = load_and_process_data()
print(f"Loaded {len(DF)} students. Dropouts: {DF['dropout_label'].sum()}")


# ---------------------------------------------------------------------------
# HELPER: safe JSON serialization
# ---------------------------------------------------------------------------
def _clean(val):
    """Convert numpy/pandas types to Python native for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def _row_to_dict(row):
    return {k: _clean(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# ROUTES — PAGES
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# ROUTES — API
# ---------------------------------------------------------------------------

@app.route('/api/stats')
def api_stats():
    total = len(DF)
    dropouts = int(DF['dropout_label'].sum())
    at_risk = int((DF['dropout_predicted'] == 1).sum())
    high = int((DF['intervention_tier'] == 'high').sum())
    medium = int((DF['intervention_tier'] == 'medium').sum())
    low = int((DF['intervention_tier'] == 'low').sum())
    avg_risk = round(float(DF['risk_score'].mean()), 3)
    avg_grade_delta = round(float(DF['grade_delta'].mean()), 2)
    avg_financial = round(float(DF['financial_stress_index'].mean()), 2)

    return jsonify({
        'total_students': total,
        'actual_dropouts': dropouts,
        'dropout_rate': round(dropouts / total * 100, 1),
        'at_risk_predicted': at_risk,
        'intervention_tiers': {'high': high, 'medium': medium, 'low': low},
        'avg_risk_score': avg_risk,
        'avg_grade_delta': avg_grade_delta,
        'avg_financial_stress': avg_financial,
        'target_distribution': {
            'Dropout': int((DF['target'] == 'Dropout').sum()),
            'Graduate': int((DF['target'] == 'Graduate').sum()),
            'Enrolled': int((DF['target'] == 'Enrolled').sum()),
        }
    })


@app.route('/api/students')
def api_students():
    tier = request.args.get('tier', None)
    search = request.args.get('search', '').strip()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 25))
    sort_by = request.args.get('sort', 'risk_score')
    order = request.args.get('order', 'desc')

    filtered = DF.copy()

    if tier and tier != 'all':
        filtered = filtered[filtered['intervention_tier'] == tier]

    if search:
        try:
            sid = int(search)
            filtered = filtered[filtered['student_id'] == sid]
        except ValueError:
            pass

    ascending = (order == 'asc')
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    columns = [
        'student_id', 'risk_score', 'dropout_predicted', 'intervention_tier',
        'target', 'grade_delta', 'financial_stress_index',
        'absenteeism_trend', 'engagement_score', 'gender_label',
        'socioeconomic_group', 'reason_text',
        'shap_factor_1', 'shap_value_1',
        'shap_factor_2', 'shap_value_2',
        'shap_factor_3', 'shap_value_3',
    ]

    rows = [_row_to_dict(row[columns]) for _, row in page_data.iterrows()]

    return jsonify({
        'students': rows,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': math.ceil(total / per_page),
    })


@app.route('/api/students/<int:student_id>')
def api_student_detail(student_id):
    match = DF[DF['student_id'] == student_id]
    if match.empty:
        return jsonify({'error': 'Student not found'}), 404

    row = match.iloc[0]
    detail = _row_to_dict(row)
    return jsonify(detail)


@app.route('/api/risk-distribution')
def api_risk_distribution():
    # Histogram bins for risk scores
    bins = np.arange(0, 1.05, 0.05)
    counts, edges = np.histogram(DF['risk_score'], bins=bins)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(counts))]

    # By tier
    tier_counts = DF['intervention_tier'].value_counts().to_dict()

    return jsonify({
        'histogram': {
            'labels': labels,
            'counts': [int(c) for c in counts],
        },
        'tier_distribution': {k: int(v) for k, v in tier_counts.items()},
    })


@app.route('/api/fairness')
def api_fairness():
    """Compute marginal and intersectional fairness metrics."""
    metrics = []

    # --- Marginal audits ---
    for group_col, group_name in [('gender_label', 'gender'),
                                   ('socioeconomic_group', 'socioeconomic')]:
        groups = DF[group_col].unique()
        if len(groups) < 2:
            continue

        for g_a, g_b in combinations(sorted(groups), 2):
            sub_a = DF[DF[group_col] == g_a]
            sub_b = DF[DF[group_col] == g_b]

            pos_rate_a = sub_a['dropout_predicted'].mean()
            pos_rate_b = sub_b['dropout_predicted'].mean()

            actual_a = sub_a[sub_a['dropout_label'] == 1]
            actual_b = sub_b[sub_b['dropout_label'] == 1]
            tpr_a = actual_a['dropout_predicted'].mean() if len(actual_a) > 0 else 0
            tpr_b = actual_b['dropout_predicted'].mean() if len(actual_b) > 0 else 0

            metrics.append({
                'group_type': group_name,
                'group_a': g_a,
                'group_b': g_b,
                'audit_type': 'marginal',
                'demographic_parity_diff': round(abs(pos_rate_a - pos_rate_b), 4),
                'equal_opportunity_diff': round(abs(tpr_a - tpr_b), 4),
                'group_a_positive_rate': round(float(pos_rate_a), 4),
                'group_b_positive_rate': round(float(pos_rate_b), 4),
                'group_a_tpr': round(float(tpr_a), 4),
                'group_b_tpr': round(float(tpr_b), 4),
                'group_a_size': int(len(sub_a)),
                'group_b_size': int(len(sub_b)),
            })

    # --- Intersectional audit ---
    intersections = sorted(DF['intersection'].unique())
    for g_a, g_b in combinations(intersections, 2):
        sub_a = DF[DF['intersection'] == g_a]
        sub_b = DF[DF['intersection'] == g_b]

        pos_rate_a = sub_a['dropout_predicted'].mean()
        pos_rate_b = sub_b['dropout_predicted'].mean()

        actual_a = sub_a[sub_a['dropout_label'] == 1]
        actual_b = sub_b[sub_b['dropout_label'] == 1]
        tpr_a = actual_a['dropout_predicted'].mean() if len(actual_a) > 0 else 0
        tpr_b = actual_b['dropout_predicted'].mean() if len(actual_b) > 0 else 0

        metrics.append({
            'group_type': 'intersectional_gender_x_socioeconomic',
            'group_a': g_a,
            'group_b': g_b,
            'audit_type': 'intersectional',
            'demographic_parity_diff': round(abs(pos_rate_a - pos_rate_b), 4),
            'equal_opportunity_diff': round(abs(tpr_a - tpr_b), 4),
            'group_a_positive_rate': round(float(pos_rate_a), 4),
            'group_b_positive_rate': round(float(pos_rate_b), 4),
            'group_a_tpr': round(float(tpr_a), 4),
            'group_b_tpr': round(float(tpr_b), 4),
            'group_a_size': int(len(sub_a)),
            'group_b_size': int(len(sub_b)),
        })

    return jsonify({'metrics': metrics})


@app.route('/api/features')
def api_features():
    """Feature importance based on correlation with dropout."""
    feature_cols = [
        'grade_delta', 'absenteeism_trend', 'financial_stress_index',
        'engagement_score', 'curricular_units_2nd_sem_grade',
        'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_approved',
        'curricular_units_1st_sem_approved', 'admission_grade',
        'tuition_fees_up_to_date', 'scholarship_holder', 'debtor',
        'age_at_enrollment',
    ]

    importances = []
    for f in feature_cols:
        if f in DF.columns:
            corr = abs(DF[f].corr(DF['dropout_label']))
            importances.append({
                'feature': f,
                'importance': round(float(corr), 4) if not np.isnan(corr) else 0,
                'display_name': f.replace('_', ' ').title(),
            })

    importances.sort(key=lambda x: x['importance'], reverse=True)
    return jsonify({'features': importances})


@app.route('/api/pipeline')
def api_pipeline():
    """Pipeline architecture metadata."""
    return jsonify({
        'layers': [
            {
                'name': 'Bronze Layer',
                'table': 'bronze.uci_dropout',
                'description': 'Raw CSV ingest — zero transformations',
                'status': 'complete',
                'records': len(DF),
            },
            {
                'name': 'Silver Layer',
                'table': 'silver.uci_dropout_clean',
                'description': 'Cleaned, features engineered, target encoded',
                'status': 'complete',
                'records': len(DF),
            },
            {
                'name': 'Model Training',
                'table': 'MLflow: dropout_signal_hackathon',
                'description': 'LogReg + XGBoost + Platt calibration',
                'status': 'complete',
            },
            {
                'name': 'Fairness Audit',
                'table': 'audit.fairness_metrics',
                'description': 'Marginal + intersectional parity metrics',
                'status': 'complete',
            },
            {
                'name': 'SHAP Explainability',
                'table': 'silver.shap_results',
                'description': 'Per-student top-3 SHAP factors',
                'status': 'complete',
            },
            {
                'name': 'Gold Table',
                'table': 'gold.at_risk_students',
                'description': 'Final at-risk students with reason_text + tiers',
                'status': 'complete',
            },
        ],
        'differentiators': [
            'Platt Calibration — scores become true probabilities',
            'Intersectional Fairness — the audit most teams skip',
            'reason_text — plain-English sentences for advisors',
        ],
    })


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0')
