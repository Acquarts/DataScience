# 🌿 About Dataset

This dataset simulates real-world smart farming operations powered by IoT sensors and satellite data. It captures environmental and operational variables that affect crop yield across **500 farms** located in regions like **India, the USA, and Africa**.

Designed to reflect modern **agritech systems**, the data is ideal for:

- Predictive modeling using ML/AI  
- Time-series analysis  
- Sensor-based optimization  
- Environmental data visualizations  
- Crop health analytics  

---

## 🧠 Ideal For

- Supervised ML models (regression, classification)  
- Yield prediction and optimization  
- Agricultural decision support systems  
- Smart irrigation strategy analysis  
- Data visualization of regional farm efficiency  

---

## 📌 Columns Description

| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `farm_id`                | Unique ID for each smart farm (e.g., FARM0001)                              |
| `region`                 | Geographic region (e.g., North India, South USA)                            |
| `crop_type`              | Crop grown: Wheat, Rice, Maize, Cotton, Soybean                             |
| `soil_moisture_%`        | Soil moisture content in percentage                                          |
| `soil_pH`                | Soil pH level (5.5–7.5 typical range)                                        |
| `temperature_C`          | Average temperature during crop cycle (in °C)                                |
| `rainfall_mm`            | Total rainfall received in mm                                               |
| `humidity_%`             | Average humidity level in percentage                                         |
| `sunlight_hours`         | Average sunlight hours received per day                                     |
| `irrigation_type`        | Type of irrigation: Drip, Sprinkler, Manual, None                            |
| `fertilizer_type`        | Fertilizer used: Organic, Inorganic, Mixed                                   |
| `pesticide_usage_ml`     | Daily pesticide usage in milliliters                                        |
| `sowing_date`            | Date when crop was sown                                                     |
| `harvest_date`           | Date when crop was harvested                                                |
| `total_days`             | Crop growth duration (harvest - sowing)                                     |
| `yield_kg_per_hectare`   | 🌾 **Target variable**: Crop yield in kilograms per hectare                  |
| `sensor_id`              | ID of the IoT sensor reporting the data                                     |
| `timestamp`              | Random in-cycle timestamp when the data snapshot was recorded               |
| `latitude`               | Farm location latitude (10.0 - 35.0 range)                                   |
| `longitude`              | Farm location longitude (70.0 - 90.0 range)                                  |
| `NDVI_index`             | Normalized Difference Vegetation Index (0.3 - 0.9)                           |
| `crop_disease_status`    | Crop disease status: None, Mild, Moderate, Severe                           |

