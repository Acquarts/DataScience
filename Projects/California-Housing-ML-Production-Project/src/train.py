import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Si no tienes xgboost instalado o prefieres no usarlo, comenta estas dos líneas y el modelo
from xgboost import XGBRegressor

TARGET = "median_house_value"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    # 1) Datos
    df = pd.read_csv("data/housing.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2) Columnas
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # 3) Preprocesador
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_features),
                             ("cat", cat_pipe, cat_features)])

    # 4) Modelos candidatos
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_estimators=400
        ),
        "XGBRegressor": XGBRegressor(
            random_state=RANDOM_STATE, n_estimators=600, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, max_depth=6, tree_method="hist",
            n_jobs=-1,
        ),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    best_name, best_rmse, best_pipe = None, float("inf"), None

    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        mse = mean_squared_error(y_te, preds)   # ← compat
        rmse = np.sqrt(mse)                     # ← compat
        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)

        print(f"{name:16s}  MAE={mae:9.2f}  RMSE={rmse:9.2f}  R²={r2:6.3f}")

        if rmse < best_rmse:
            best_name, best_rmse, best_pipe = name, rmse, pipe

    joblib.dump(best_pipe, "artifacts/model.joblib")
    print(f"\n✅ Modelo guardado en artifacts/model.joblib  (mejor: {best_name}, RMSE={best_rmse:.2f})")

if __name__ == "__main__":
    main()
