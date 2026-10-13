"""
train_model.py
Generates data, trains several regression models, compares them,
saves the best one, and plots results.
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # works without a display; remove if you want pop-up plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from generate_data import generate_students

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours",
    "assignments_done",
    "extra_activities",
]
TARGET = "final_marks"


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:20s} MAE={mae:5.2f}  RMSE={rmse:5.2f}  R2={r2:.3f}")
    return r2, pred


def main():
    df = generate_students(n_samples=1000)
    df.to_csv("students_synthetic.csv", index=False)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    print("Model comparison (test set):")
    best_name, best_model, best_r2, best_pred = None, None, -np.inf, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2, pred = evaluate(name, model, X_test, y_test)
        if r2 > best_r2:
            best_name, best_model, best_r2, best_pred = name, model, r2, pred

    print(f"\nBest model: {best_name} (R2={best_r2:.3f})")
    joblib.dump({"model": best_model, "features": FEATURES}, "marks_model.joblib")
    print("Saved model to marks_model.joblib")

    # Plot 1: actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], "r--")
    plt.xlabel("Actual marks")
    plt.ylabel("Predicted marks")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    plt.close()

    # Plot 2: feature importance (or coefficients)
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        label = "Importance"
    else:
        importance = np.abs(best_model.coef_)
        label = "|Coefficient|"
    order = np.argsort(importance)
    plt.figure(figsize=(7, 4))
    plt.barh(np.array(FEATURES)[order], importance[order])
    plt.xlabel(label)
    plt.title("What affects marks the most?")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("Saved plots: actual_vs_predicted.png, feature_importance.png")


if __name__ == "__main__":
    main()