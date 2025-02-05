import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


def read_data(dataset_path):
    """Reads a CSV dataset and returns a DataFrame."""
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        return None


def preprocess_data(df, target_column, scaler_type):
    """Splits the data into train-test and applies scaling."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler type")

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model, model_name):
    """Trains a model and saves it to a file."""
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, model_path  # Return model and file path for download


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and returns accuracy."""
    return model.score(X_test, y_test)