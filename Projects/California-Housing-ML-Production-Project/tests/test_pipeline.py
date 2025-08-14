import pandas as pd
from src.data import load_data
from src.pipeline import build_preprocessing_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def test_pipeline_runs():
    df = load_data()
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    num_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = build_preprocessing_pipeline(num_features, cat_features)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)
