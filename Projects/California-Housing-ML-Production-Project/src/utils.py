import joblib

def save_model(model, path="artifacts/model.joblib"):
    """Guarda el modelo entrenado en la ruta indicada."""
    joblib.dump(model, path)

def load_model(path="artifacts/model.joblib"):
    """Carga el modelo entrenado desde la ruta indicada."""
    return joblib.load(path)
