"""
generate_data.py
Creates a synthetic student dataset (no external dataset needed).

Final marks are generated from a realistic formula plus random noise, so the
ML model has a real pattern to learn:
    marks = f(study hours, attendance, previous marks, sleep, assignments, ...) + noise
"""

import numpy as np
import pandas as pd


def generate_students(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(4, 2, n_samples), 0, 12)          # hours/day
    attendance = np.clip(rng.normal(75, 15, n_samples), 20, 100)       # percent
    previous_marks = np.clip(rng.normal(60, 15, n_samples), 10, 100)   # percent
    sleep_hours = np.clip(rng.normal(7, 1.2, n_samples), 3, 10)        # hours/day
    assignments_done = np.clip(rng.normal(75, 20, n_samples), 0, 100)  # percent
    extra_activities = rng.integers(0, 4, n_samples)                   # clubs/sports count

    noise = rng.normal(0, 4, n_samples)

    # Sleep helps up to ~8 hours; too little or too much hurts slightly.
    sleep_effect = -1.5 * np.abs(sleep_hours - 8)

    marks = (
        8
        + 3.0 * study_hours
        + 0.18 * attendance
        + 0.40 * previous_marks
        + 0.12 * assignments_done
        + sleep_effect
        - 0.8 * np.maximum(extra_activities - 2, 0)
        + noise
    )
    marks = np.clip(marks, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(1),
            "attendance": attendance.round(1),
            "previous_marks": previous_marks.round(1),
            "sleep_hours": sleep_hours.round(1),
            "assignments_done": assignments_done.round(1),
            "extra_activities": extra_activities,
            "final_marks": marks.round(1),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_students()
    df.to_csv("students_synthetic.csv", index=False)
    print(f"Saved students_synthetic.csv with {len(df)} rows")
    print(df.head())