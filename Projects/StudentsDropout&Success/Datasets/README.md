## 📘 About the Dataset

This dataset originates from a Portuguese higher education institution and was developed as part of a national project aiming to combat student dropout and academic failure in universities.  
It contains information from **4,424 undergraduate students** across **8 degree programs**, including:

- Agronomy
- Design
- Education
- Nursing
- Journalism
- Management
- Social Service
- Technologies

---

## 🎯 Objective

The main goal is to enable **early intervention** by using machine learning models to predict a student’s academic outcome:

- **Drop out**
- **Remain Enrolled**
- **Successfully Graduate**

This is a **three-class classification problem** with a known class imbalance, offering realistic challenges for predictive modeling and education analytics.

---

## 📊 Dataset Highlights

- **Instances**: 4,424 students  
- **Features**: 36 total  
- **Types**: Integer, Categorical, Real-valued  
- Includes **demographic and academic** information  
- **Target Variable**: `'Target'` (Categorical)  
  - Classes: `Dropout`, `Enrolled`, `Graduate`

---

## 🧩 Feature Categories

### 📌 Demographics & Socioeconomic:
- Gender, Age, Marital Status
- Nationality
- Parental Education and Occupation
- Scholarship holder
- Tuition fees
- Application mode

### 📌 Academic History:
- Degree program
- Curricular units enrolled & approved
- Grades (1st and 2nd semesters)
- Admission grade
- Previous qualification

### 📌 External Factors:
- GDP at enrollment time
- Inflation rate at enrollment time

---

## 🧼 Data Preprocessing

The original researchers performed extensive cleaning and handling of:

- Outliers  
- Inconsistent entries  
- Anomalies  
- Missing values  

✅ **Final dataset contains no missing values**.

---

## 💡 Suggested Use Cases

- Educational Data Mining  
- Early Warning Systems for Student Dropout  
- Classification Benchmarking  
- Feature Importance & Interpretability Studies  
- Policy-making simulations for academic retention

---

## ⚙️ Recommended Setup

- **Task Type**: Multiclass Classification  
- **Evaluation Metrics**: Accuracy, Macro F1 Score, Confusion Matrix  
- **Suggested Split**: 80% Training / 20% Testing

---

## 📚 Citation & Source

This dataset was created under the **SATDAP - Capacitação da Administração Pública** project, funded by **POCI-05-5762-FSE-000191 (Portugal)**, and is available through the **UCI Machine Learning Repository**.

