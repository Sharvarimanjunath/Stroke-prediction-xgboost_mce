
Intelligent Health Care System for Stroke Prediction Using Machine Learning

Project Overview

This project develops a machine learning-based healthcare system to predict the risk of stroke using patient health and demographic information. It uses XGBoost as the final prediction model and provides an interactive Streamlit application.

Technologies Used

- Python
- XGBoost
- Balanced Random Forest
- Pandas and NumPy
- Scikit-learn
- SMOTE
- SHAP
- Streamlit
- Matplotlib and Seaborn

Key Features

- Patient data preprocessing
- Handling imbalanced datasets
- Machine learning model comparison
- XGBoost-based stroke prediction
- Hyperparameter tuning
- Decision threshold optimization
- Model performance evaluation
- SHAP-based explainability
- Interactive Streamlit interface

Project Structure

- "data/" – Dataset
- "models/" – Trained models, evaluation results and visualizations
- "train_model_xgboost.py" – Model training and evaluation
- "app_xgboost.py" – Streamlit application

How to Run

Install the required packages:

pip install pandas numpy scikit-learn xgboost imbalanced-learn shap streamlit matplotlib seaborn joblib

Run model training:

python train_model_xgboost.py

Launch the application:

streamlit run app_xgboost.py

Explainability

SHAP (SHapley Additive exPlanations) is used to explain the contribution of input features to model predictions.

Disclaimer

This project is developed for academic and educational purposes. It is not intended to replace professional medical diagnosis.

Author

Sharvari Manjunath
