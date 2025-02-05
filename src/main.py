import os
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

from ml_utility import read_data, preprocess_data, train_model, evaluate_model

# Get the parent directory
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

st.set_page_config(page_title="Automate ML", page_icon="🧠", layout="centered")

st.title("🤖 No Code ML Model Training")

dataset_list = os.listdir(f"{parent_dir}/data")

dataset = st.selectbox("Select a dataset from the dropdown", dataset_list, index=None)

if dataset:
    df = read_data(os.path.join(parent_dir, "data", dataset))  # FIXED

    if df is not None:
        st.dataframe(df.head())  # Display dataset

        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier()
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name", "my_model")

        if st.button("Train the Model"):
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            trained_model, model_path = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy = evaluate_model(trained_model, X_test, y_test)

            st.success(f"Test Accuracy: {accuracy:.2f}")

            with open(model_path, "rb") as f:
                st.download_button(label="Download Model", data=f, file_name=f"{model_name}.pkl", mime="application/octet-stream")