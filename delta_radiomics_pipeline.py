# delta_radiomics_pipeline.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import radiomics
from radiomics import featureextractor

# ===================== CONFIGURATION =====================
DATA_DIR = 'data'
LABELS_FILE = 'data/labels.csv'
OUTPUT_FEATURES = 'all_features.csv'
OUTPUT_DELTA = 'delta_features.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# PyRadiomics parameters (IBSI compliant)
PARAMS = {
    'binWidth': 0.1,
    'resampledPixelSpacing': [2, 2, 2],
    'interpolator': 'sitkBSpline',
    'normalize': False,
    'force2D': False,
    'verbose': False
}

# ===================== STEP 1: FEATURE EXTRACTION =====================
extractor = featureextractor.RadiomicsFeatureExtractor(**PARAMS)
# Enable all feature classes (or select specific ones)
extractor.enableAllFeatures()

def extract_features(patient_dir, timepoint):
    pet = os.path.join(patient_dir, timepoint, 'PET.nii.gz')
    mask = os.path.join(patient_dir, timepoint, 'mask.nii.gz')
    if not os.path.exists(pet) or not os.path.exists(mask):
        return None
    result = extractor.execute(pet, mask)
    feats = {f"{timepoint}_{k}": v for k, v in result.items() if isinstance(v, (int, float))}
    return feats

# Get patient list
patients = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('patient')]
labels = pd.read_csv(LABELS_FILE, index_col='patient_id')

all_data = []
for pid in patients:
    pat_dir = os.path.join(DATA_DIR, pid)
    f0 = extract_features(pat_dir, 'T0')
    f1 = extract_features(pat_dir, 'T1')
    if f0 is None or f1 is None:
        continue
    combined = {**f0, **f1}
    combined['patient_id'] = pid
    combined['response'] = labels.loc[pid, 'response']
    # Optionally add center info
    # combined['center'] = labels.loc[pid, 'center']
    all_data.append(combined)

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_FEATURES, index=False)

# ===================== STEP 2: DELTA FEATURE CALCULATION =====================
df = pd.read_csv(OUTPUT_FEATURES)
feature_cols = [c for c in df.columns if c.startswith('T0_') or c.startswith('T1_')]
for col in feature_cols:
    if col.startswith('T0_'):
        base = col[3:]
        t1c = 'T1_' + base
        if t1c in df.columns:
            delta = (df[t1c] - df[col]) / (df[col] + 1e-6)
            df['delta_' + base] = delta

delta_cols = [c for c in df.columns if c.startswith('delta_')]
df_delta = df[['patient_id', 'response'] + delta_cols].dropna()
df_delta.to_csv(OUTPUT_DELTA, index=False)

# ===================== STEP 3: FEATURE SELECTION (LASSO) =====================
X = df_delta[delta_cols].values
y = df_delta['response'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_selector = LogisticRegressionCV(Cs=10, cv=CV_FOLDS, penalty='l1', solver='liblinear', random_state=RANDOM_STATE)
lasso_selector.fit(X_scaled, y)
selected = [delta_cols[i] for i, coef in enumerate(lasso_selector.coef_[0]) if abs(coef) > 1e-3]
print(f"Selected {len(selected)} features: {selected}")

# ===================== STEP 4: TRAIN-TEST SPLIT =====================
X_sel = df_delta[selected].values
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

# ===================== STEP 5: MODEL TRAINING =====================
models = {
    'LR': LogisticRegression(random_state=RANDOM_STATE),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE),
    'RF': RandomForestClassifier(random_state=RANDOM_STATE),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
}

param_grids = {
    'LR': {'C': [0.01, 0.1, 1, 10]},
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'RF': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'XGB': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
}

best_models = {}
for name in models:
    print(f"\n=== Training {name} ===")
    gs = GridSearchCV(models[name], param_grids[name], cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    best_models[name] = gs.best_estimator_
    print(f"Best CV AUC: {gs.best_score_:.3f}")
    # Test evaluation
    y_prob = gs.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.3f}")
