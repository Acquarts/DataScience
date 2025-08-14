import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "median_house_value"

def main():
    df = pd.read_csv("data/housing.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    model = joblib.load("artifacts/model.joblib")
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)  # compat
    rmse = np.sqrt(mse)                 # compat
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"Evaluación completa →  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

if __name__ == "__main__":
    main()
