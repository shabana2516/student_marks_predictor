"""
predict.py
Loads the trained model and predicts a student's final marks from
values typed in by the user. Run train_model.py first.
"""

import sys

import joblib
import pandas as pd

PROMPTS = {
    "study_hours": ("Study hours per day (0-12)", 0, 12),
    "attendance": ("Attendance % (0-100)", 0, 100),
    "previous_marks": ("Previous exam marks % (0-100)", 0, 100),
    "sleep_hours": ("Sleep hours per day (3-10)", 3, 10),
    "assignments_done": ("Assignments completed % (0-100)", 0, 100),
    "extra_activities": ("Number of extra activities/clubs (0-3)", 0, 3),
}


def ask(label, low, high):
    while True:
        try:
            value = float(input(f"{label}: "))
            if low <= value <= high:
                return value
            print(f"  Please enter a value between {low} and {high}.")
        except ValueError:
            print("  Please enter a number.")


def grade(marks):
    if marks >= 90:
        return "A+"
    if marks >= 80:
        return "A"
    if marks >= 70:
        return "B"
    if marks >= 60:
        return "C"
    if marks >= 50:
        return "D"
    return "F"


def main():
    try:
        saved = joblib.load("marks_model.joblib")
    except FileNotFoundError:
        print("Model not found. Run: python train_model.py")
        sys.exit(1)

    model, features = saved["model"], saved["features"]

    print("=== Student Marks Predictor ===")
    values = {f: ask(*PROMPTS[f]) for f in features}

    X = pd.DataFrame([values], columns=features)
    marks = float(model.predict(X)[0])
    marks = max(0.0, min(100.0, marks))

    print(f"\nPredicted final marks: {marks:.1f} / 100  (Grade {grade(marks)})")


if __name__ == "__main__":
    main()