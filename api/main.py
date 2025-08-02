# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# Load the model
model = mlflow.sklearn.load_model("mlruns/<RUN_ID>/artifacts/model")

# Create FastAPI app
app = FastAPI(title="Customer Churn Predictor")

# Define input schema
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: ChurnInput):
    # Convert input to dataframe
    input_df = pd.DataFrame([data.dict()])

    # Apply same preprocessing as training (you must match the encoded form!)
    # Ideally, you'd load and apply the same label encoders here
    # For this example, assume preprocessing already handled elsewhere

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(probability, 3)
    }
