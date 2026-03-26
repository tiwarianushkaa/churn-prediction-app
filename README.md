# Customer Churn Prediction App

## Live Application
https://churn-prediction-app-2eeutfkdtjvzsfccgfrtmy.streamlit.app/

## Overview
This project is an end-to-end machine learning application designed to predict customer churn using historical customer data. It covers the complete data science workflow including data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive web application.

The application allows users to upload a dataset, analyze customer churn patterns, and make real-time predictions using a trained machine learning model.

## Features
- End-to-end data processing pipeline
- Automatic detection of target variable (Churn / Exited / Target)
- Data cleaning, handling missing values, and encoding
- Machine learning model using Random Forest
- Model evaluation with accuracy and classification report
- Interactive Streamlit web interface
- Real-time prediction with probability score
- Dynamic feature handling for new input data

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Structure
churn-prediction/
│── app.py
│── model.py
│── preprocess.py
│── requirements.txt
│── README.md

## How It Works
1. User uploads a CSV dataset
2. Data is cleaned and preprocessed automatically
3. Features are encoded using one-hot encoding
4. Model is trained using Random Forest Classifier
5. Performance metrics are displayed
6. User can input new customer data for prediction
7. Model outputs churn prediction with probability

## Installation

Clone the repository:
git clone https://github.com/tiwarianushkaa/churn-prediction-app.git
cd churn-prediction-app

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

## Input Requirements
- CSV file containing customer data
- Dataset must include a target column such as:
  - Churn
  - Exited
  - Target

## Model Details
- Algorithm: Random Forest Classifier
- Train/Test Split: 80/20
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Future Improvements
- Hyperparameter tuning
- Model persistence (save/load trained model)
- Advanced visualizations
- Support for multiple datasets
- Improved feature engineering

## License
This project is developed for educational and demonstration purposes.
