# train_model.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# MLflow setup
mlflow.set_tracking_uri("file:///C:/Users/NANOTEK/Desktop/mlops freelance/mlruns")
mlflow.set_experiment("Telco Churn Prediction")

with mlflow.start_run():
    # -----------------------------
    # Hyperparameter Grid
    # -----------------------------
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    # -----------------------------
    # Model & GridSearchCV
    # -----------------------------
    base_model = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(base_model, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # -----------------------------
    # Evaluate on Test Set
    # -----------------------------
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # -----------------------------
    # MLflow Logging
    # -----------------------------
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Feature Importance Plot
    feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print(f"✅ Best Model Trained - Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
