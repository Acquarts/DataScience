# Thyroid Cancer Recurrence Analysis and Prediction

This project analyzes clinical data from thyroid cancer patients to predict recurrence after radioactive iodine (RAI) therapy. It explores patterns, answers key clinical questions, and trains machine learning models for predictive purposes.

## 📁 Dataset

- **Records:** 383 patients
- **Features:** 13 clinical attributes
- **Target:** `Recurred` variable (Yes / No)
- **No missing values**

## ❓ Key Questions

1. Are thyroid cancer recurrences more common in men or women?
2. How does age affect recurrence risk?
3. Can we predict recurrence based on tumor staging and pathology?
4. What is the relationship between treatment response and recurrence?

## 🔍 Exploratory Data Analysis

- Distributions by gender, age, risk level, stage, pathology, and treatment response.
- Recurrence proportions for each categorical variable.
- Correlation matrix to identify the most predictive features.

## 🧠 Trained Models

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

### 🎯 Comparative Metrics (Macro Avg)

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.91     | 0.88      | 0.91   | 0.89     |
| Random Forest       | 0.90     | 0.88      | 0.86   | 0.87     |
| XGBoost             | 0.86     | 0.82      | 0.83   | 0.83     |

## ✅ Conclusions

- Recurrences are more frequent in **men**.
- **Age** shows a moderate relationship with recurrence.
- Models can effectively predict recurrence using **tumor stage, response, and risk classification**.
- `Response` is the variable with the **strongest positive correlation (0.71)** to recurrence.
- `Risk` has a **strong negative correlation (-0.73)** and is also a powerful predictor.

## 📌 Recommendations

This model can serve as a useful tool for **risk stratification** and **personalized clinical follow-up**.

**Author:** Adrian Zambrana · April 2025

---

