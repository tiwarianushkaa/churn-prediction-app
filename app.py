import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from model import train_model

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.write("Upload your dataset to train the model and generate insights.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("🔍 Raw Data")
    st.write(df.head())

    # Preprocess
    df_processed = preprocess_data(df)

    st.subheader("⚙️ Processed Data")
    st.write(df_processed.head())

    # Train model
    model, acc, report = train_model(df_processed)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Total Records", len(df))

    st.subheader("📄 Classification Report")
    st.text(report)

    # Chart
    if 'Churn' in df.columns:
        st.subheader("📊 Churn Distribution")
        st.bar_chart(df['Churn'].value_counts())

    st.success("✅ Model trained successfully!")

    # ---------------- PREDICTION UI ---------------- #
    st.subheader("🔮 Predict Customer Churn")

    # User Inputs (clean UI)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict"):

        # Create input row
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet
        }

        input_df = pd.DataFrame([input_dict])

        # Combine with original data (important for encoding match)
        combined = pd.concat([df, input_df], ignore_index=True)

        # Apply preprocessing
        combined_processed = preprocess_data(combined)

        # Extract last row (user input)
        input_processed = combined_processed.tail(1)

        # Align columns with training data
        target_col = [col for col in df_processed.columns if 'churn' in col.lower()][0]
        model_features = df_processed.drop(columns=[target_col]).columns

        input_processed = input_processed.reindex(columns=model_features, fill_value=0)

        # Predict
        prediction = model.predict(input_processed)
        probability = model.predict_proba(input_processed)[0][1]

        if prediction[0] == 1:
            st.error(f"❌ Customer likely to churn ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Customer likely to stay ({(1-probability)*100:.2f}%)")

else:
    st.warning("⬆️ Please upload a CSV file to continue.")