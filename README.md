# Student Marks Predictor (AI/ML Project)

Predicts a student's final marks from study habits and academic history using
machine learning regression. **No external dataset is needed** - the project
generates its own synthetic data.

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates the synthetic student dataset (1000 rows) |
| `train_model.py` | Trains Linear Regression and Random Forest, compares them, saves the best, makes plots |
| `predict.py` | Asks for a student's details and predicts final marks |
| `requirements.txt` | Python libraries needed |

## How to run

```bash
pip install -r requirements.txt
python train_model.py     # trains the model and creates plots
python predict.py         # interactive prediction
```

## Input features

study_hours, attendance, previous_marks, sleep_hours, assignments_done, extra_activities

## Output

Predicted final marks (0-100) and a letter grade.

## How it works

1. Synthetic data is generated from a formula (study hours, attendance and
   previous marks raise marks; very low/high sleep and too many activities lower
   them) plus random noise.
2. Data is split 80% train / 20% test.
3. Linear Regression and Random Forest are trained and compared using MAE,
   RMSE and R2.
4. The best model is saved with joblib and used by `predict.py`.

## Note for report/viva

The data is synthetic, so results show the ML pipeline works, not real-world
accuracy. To use real data, replace the DataFrame in `train_model.py` with
`pd.read_csv("your_file.csv")` using the same column names.

## Ideas to extend

- Add a Streamlit or Flask web interface
- Add more models (Gradient Boosting, SVR) and cross-validation
- Replace synthetic data with a real student dataset (e.g. UCI Student Performance)