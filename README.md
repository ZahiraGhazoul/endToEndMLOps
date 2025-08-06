# 🧠 MLOps Project: End-to-End Pipeline for Predicting Customer Churn

This project demonstrates an end-to-end MLOps pipeline using the **Telco Customer Churn Dataset**. It includes data preprocessing, model training and evaluation, model serving via REST API (FastAPI), containerization (Docker), and CI/CD automation.

---

## 🚀 Project Architecture

Data Preparation → Model Training & Logging (MLflow) → Model Serving (FastAPI) → Docker Containerization → CI/CD Deployment

yaml
Copier
Modifier

---

## 📁 Project Structure

mlops-customer-churn/
│
├── data/ # Raw and processed datasets
├── model/
│ ├── train_model.py # Model training script
│ └── churn_model.pkl # Trained model
│
├── api/
│ └── main.py # FastAPI app for model inference
│
├── Dockerfile # Docker container config
├── requirements.txt # Python dependencies
├── .github/workflows/ # CI/CD workflows
├── README.md

yaml
Copier
Modifier

---

## 📊 Dataset

- **Source**: Telco Customer Churn Dataset  
- **Features**: Customer demographics, services used, account info  
- **Target**: Churn (Yes/No)

---

## 🛠️ Steps

### 🔹 1. Data Preparation

- Load data using `pandas`
- Handle missing values
- Encode categorical features
- Train/test split

### 🔹 2. Model Training & Evaluation

- Algorithm: `RandomForestClassifier` from `scikit-learn`
- Metrics: Accuracy, F1-score, AUC
- Logging: `MLflow` used to track model, parameters, and metrics

### 🔹 3. Model Serving via FastAPI

- Endpoint: `/predict`
- Accepts JSON input of customer features
- Returns predicted churn result

### 🔹 4. Containerization

- Created a `Dockerfile` to containerize the FastAPI app
- Simplifies deployment

### 🔹 5. CI/CD Pipeline (Optional)

- GitHub Actions used to automate testing, build, and deploy steps
- On each `push`, run unit tests and build Docker image

---

## 🧪 How to Run

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
🏋️‍♀️ Train the Model
bash
Copier
Modifier
python model/train_model.py
🚀 Launch FastAPI Server
bash
Copier
Modifier
uvicorn api.main:app --reload
🐳 Run with Docker
bash
Copier
Modifier
docker build -t churn-api .
docker run -p 8000:8000 churn-api
📫 API Example
POST /predict

json
Copier
Modifier
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No"
  // other features...
}
Response:

json
Copier
Modifier
{
  "prediction": "Churn"
}
✅ Tools & Tech
Python (Pandas, Scikit-learn)

MLflow

FastAPI

Docker

GitHub Actions (CI/CD)

👤 Author
Asma Ghazoul – Cloud & DevOps Engineer with a passion for MLOps
🔗 LinkedIn
