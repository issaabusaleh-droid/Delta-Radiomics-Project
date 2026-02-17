from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

models = {
    'Logistic Regression': LogisticRegression(penalty='l2', random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Hyperparameter grids (simplified)
param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
}

best_estimators = {}
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Training {name}...")
    gs = GridSearchCV(estimator=model, param_grid=param_grids[name],
                      cv=cv_outer, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    best_estimators[name] = gs.best_estimator_
    print(f"Best AUC (CV): {gs.best_score_:.3f}")
