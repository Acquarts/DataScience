# 🏥 Thyroid Cancer Recurrence Predictor

## 📋 Description

Interactive web application built with Streamlit that uses Machine Learning models to predict the probability of recurrence in thyroid cancer patients. The tool is based on real data from 383 patients and provides accurate predictions using three different machine learning algorithms.

## 🎯 Key Features

- **Real-time prediction**: Get instant predictions on recurrence risk
- **Multiple ML models**: Choose between three trained models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Interactive visualizations**: Dynamic charts showing prediction probabilities
- **Feature importance analysis**: Identify which factors have the most weight in predictions (Random Forest)
- **Intuitive interface**: Clean and user-friendly design for medical professionals

## 🔬 Models and Performance

The models have been trained on a dataset of 383 patients with the following performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|---------|----------|
| Logistic Regression | 0.922 | 0.829 | 0.922 | 0.860 |
| Random Forest | 0.935 | 0.862 | 0.935 | 0.887 |
| XGBoost | 0.948 | 0.893 | 0.948 | 0.913 |

## 📊 Input Features

The application considers the following patient characteristics:
- **Age**: Patient's age in years
- **Gender**: Male/Female
- **History of Radiotherapy**: Whether the patient has received prior radiotherapy
- **Adenopathy**: Presence of enlarged lymph nodes
- **Pathology Type**: Histological classification of the tumor
- **Focality**: Unifocal or Multifocal
- **T Stage**: Size and extent of the primary tumor
- **N Stage**: Regional lymph node involvement
- **M Stage**: Presence of distant metastasis
- **Overall Stage**: General cancer classification

## 🚀 Technologies Used

- **Python 3.8+**
- **Streamlit**: Web application framework
- **Scikit-learn**: ML models and preprocessing
- **XGBoost**: Gradient boosting algorithm
- **Pandas & NumPy**: Data manipulation
- **Plotly**: Interactive visualizations

## 💻 Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Application Locally
```bash
streamlit run app.py
```

### Required Files
The application requires the following pre-trained files:
- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `model_metrics.json`
- `categorical_mappings.json`
- `feature_names.json`

## 🌐 Live Demo

You can access the deployed application at: [Your Streamlit Cloud URL]

## 📈 Training Process

To train the models from scratch:

```bash
python train_models.py
```

This script:
1. Loads and preprocesses data from `filtered_thyroid_data.csv`
2. Applies encoding to categorical variables and feature scaling
3. Trains three different models (LR, RF, XGBoost)
4. Saves models and necessary metadata for the application

## ⚠️ Disclaimer

**Important**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with an oncologist or qualified medical professional for clinical decisions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Developed as part of my studies of IA, combining knowledge of Machine Learning, Data Science, and Deep Learning applied to the healthcare sector.

## 🙏 Acknowledgments

- Thyroid Cancer Recurrence Dataset
- Streamlit community for excellent documentation
- Professors and colleagues from the AI Master's program
