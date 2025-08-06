# prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "data"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    return df.dropna()

def encode_categoricals(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_data(X_train, X_test, y_train, y_test, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

def main():
    print("🚀 Loading data...")
    df = load_data(DATA_PATH)

    print("🧹 Cleaning data...")
    df = clean_data(df)

    print("🔁 Encoding categorical features...")
    df, _ = encode_categoricals(df)

    print("✂️ Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("📏 Scaling features...")
    X_train, scaler = scale_features(X_train)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print("💾 Saving prepared data...")
    save_data(X_train, X_test, y_train, y_test)

    print("✅ Data preparation complete!")

if __name__ == "__main__":
    main()
