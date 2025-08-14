from pydantic import BaseModel

class TrainConfig(BaseModel):
    target: str = "median_house_value"
    test_size: float = 0.2
    random_state: int = 42
    models: list[str] = ["LinearRegression", "RandomForest", "XGBRegressor"]

CFG = TrainConfig()

