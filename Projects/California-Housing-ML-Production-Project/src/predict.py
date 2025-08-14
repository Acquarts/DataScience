import pandas as pd, joblib

SAMPLE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}

def main():
    model = joblib.load("artifacts/model.joblib")
    df = pd.DataFrame([SAMPLE])
    pred = model.predict(df)[0]
    print(f"Predicted median_house_value: {pred:.2f}")

if __name__ == "__main__":
    main()
