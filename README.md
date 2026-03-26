# Customer Churn Prediction App

## Overview
This project is an end-to-end machine learning application that predicts customer churn based on historical data. It includes data preprocessing, model training, evaluation, and an interactive web interface for real-time predictions.

## Features
- Data cleaning and preprocessing pipeline
- Automatic detection and handling of target variable
- Machine learning model using Random Forest
- Model evaluation with accuracy and classification report
- Interactive web application using Streamlit
- Real-time customer churn prediction with probability score

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

## Installation

1. Clone the repository:
git clone https://github.com/tiwarianushkaa/churn-prediction-app.git
cd churn-prediction-app

2. Create and activate a virtual environment (optional but recommended):
conda create -n churn_env python=3.10
conda activate churn_env

3. Install dependencies:
pip install -r requirements.txt

## Usage

Run the Streamlit application:
streamlit run app.py

Then open the provided local URL in your browser.

## Input Requirements
- CSV file containing customer data
- Dataset should include a target column such as "Churn", "Exited", or similar

## Model Details
- Algorithm: Random Forest Classifier
- Evaluation Metrics:
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)

## Future Improvements
- Hyperparameter tuning
- Model persistence (saving/loading trained model)
- Advanced visualizations
- Support for additional datasets
- Improved feature engineering

## License
This project is for educational and demonstration purposes.
