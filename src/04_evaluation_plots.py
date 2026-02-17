from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

for name, model in best_estimators.items():
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Test AUC: {auc:.3f}, Accuracy: {acc:.3f}")
