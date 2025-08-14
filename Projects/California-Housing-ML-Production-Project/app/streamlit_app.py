# app/streamlit_app.py
import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
import joblib

# =========================
# Config UI
# =========================
st.set_page_config(page_title="California Housing – Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California Housing — Predictor")
st.caption("Predicción individual vía API y por lotes cargando el modelo local")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# =========================
# Defaults desde el dataset (para prellenar y validar)
# =========================
def load_defaults():
    try:
        df = pd.read_csv("data/housing.csv")
        num_cols = [
            "longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
            "population","households","median_income"
        ]
        med = df[num_cols].median()
        p01 = df[num_cols].quantile(0.01)
        p99 = df[num_cols].quantile(0.99)
        default_ocean = df["ocean_proximity"].mode().iat[0]
        oceans = sorted(df["ocean_proximity"].unique().tolist())
        return med, p01, p99, default_ocean, oceans
    except Exception:
        # Fallback si no está el CSV
        med = pd.Series({
            "longitude": -119.5,
            "latitude": 36.5,
            "housing_median_age": 29.0,
            "total_rooms": 2127.0,
            "total_bedrooms": 435.0,
            "population": 1166.0,
            "households": 409.0,
            "median_income": 3.53
        })
        p01 = med * 0.5
        p99 = med * 1.5
        default_ocean = "INLAND"
        oceans = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        st.info("No se encontró data/housing.csv. Usando valores por defecto.")
        return med, p01, p99, default_ocean, oceans

med, p01, p99, default_ocean, oceans = load_defaults()

# =========================
# 1) Predicción individual (vía API)
# =========================
st.subheader("🔹 Predicción individual (API)")

with st.form("form_single"):
    c1, c2 = st.columns(2)
    longitude = c1.number_input(
        "Longitude",
        value=float(med["longitude"]),
        min_value=float(p01["longitude"]), max_value=float(p99["longitude"]),
        help="Longitud del distrito."
    )
    latitude = c2.number_input(
        "Latitude",
        value=float(med["latitude"]),
        min_value=float(p01["latitude"]), max_value=float(p99["latitude"]),
        help="Latitud del distrito."
    )

    c3, c4 = st.columns(2)
    housing_median_age = c3.number_input(
        "Housing median age (años, distrito)",
        value=float(med["housing_median_age"]), min_value=0.0,
        max_value=float(max(p99["housing_median_age"], med["housing_median_age"])),
        help="Edad mediana de las viviendas del distrito."
    )
    median_income = c4.number_input(
        "Median income (×10k USD, distrito)",
        value=float(med["median_income"]), min_value=0.0,
        max_value=float(max(p99["median_income"], med["median_income"] * 2)),
        format="%.4f",
        help="Ej: 8.3 ≈ 83.000 USD."
    )

    total_rooms = st.number_input(
        "Total rooms (distrito)",
        value=float(med["total_rooms"]),
        min_value=float(max(0.0, p01["total_rooms"])),
        max_value=float(max(p99["total_rooms"], med["total_rooms"] * 2)),
        step=10.0,
        help="Suma de habitaciones en el distrito (no por vivienda)."
    )
    total_bedrooms = st.number_input(
        "Total bedrooms (distrito)",
        value=float(med["total_bedrooms"]),
        min_value=float(max(0.0, p01["total_bedrooms"])),
        max_value=float(max(p99["total_bedrooms"], med["total_bedrooms"] * 2)),
        step=1.0,
        help="Suma de dormitorios en el distrito."
    )
    population = st.number_input(
        "Population (distrito)",
        value=float(med["population"]),
        min_value=float(max(0.0, p01["population"])),
        max_value=float(max(p99["population"], med["population"] * 2)),
        step=1.0
    )
    households = st.number_input(
        "Households (distrito)",
        value=float(med["households"]),
        min_value=float(max(0.0, p01["households"])),
        max_value=float(max(p99["households"], med["households"] * 2)),
        step=1.0
    )

    ocean_proximity = st.selectbox(
        "Ocean proximity",
        options=oceans,
        index=oceans.index(default_ocean) if default_ocean in oceans else 0,
        help="Categoría del distrito respecto a la costa."
    )

    submit_single = st.form_submit_button("Predecir (API)")

if submit_single:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        yhat = resp.json().get("predicted_price")
        st.success(f"Precio estimado: ${yhat:,.2f}")
        with st.expander("Payload enviado"):
            st.json(payload)
        with st.expander("Request URL"):
            st.code(API_URL)
    except requests.exceptions.RequestException as e:
        st.error(f"Error llamando a la API: {e}")
        st.info(f"¿API levantada? Ejecuta:  uvicorn api.app:app --reload")

st.markdown("---")

# =========================
# 2) Predicción por lotes (CSV) — usa el modelo local
# =========================
st.subheader("📦 Predicción por lotes (sube un CSV)")

uploaded = st.file_uploader(
    "Sube un CSV con estas columnas:",
    type=["csv"],
    help="""Columnas requeridas:
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity"""
)

required_cols = [
    "longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
    "population","households","median_income","ocean_proximity"
]

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        missing = [c for c in required_cols if c not in df_in.columns]
        if missing:
            st.error(f"Faltan columnas en tu CSV: {missing}")
        else:
            # Cargar modelo del disco y predecir
            model = joblib.load("artifacts/model.joblib")
            preds = model.predict(df_in[required_cols])
            df_out = df_in.copy()
            df_out["predicted_price"] = preds

            st.success(f"Predicciones generadas: {len(df_out)} filas")
            st.dataframe(df_out.head(20))

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar CSV con predicciones",
                data=csv_bytes,
                file_name="predicciones_california.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error procesando el CSV: {e}")
