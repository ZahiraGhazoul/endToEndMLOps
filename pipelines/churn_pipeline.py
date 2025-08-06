from kfp.dsl import component, pipeline, Output, OutputPath
from kfp import compiler

@component(base_image="python:3.9")
def train_model(model_path: OutputPath(str)):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import os

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    data = pd.read_csv('data/cleaned_telco.csv')
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, model_path)

@pipeline(
    name='churn-pipeline',
    description='Pipeline to train a churn prediction model.'
)
def churn_pipeline():
    train_model()

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path='churn_pipeline.yaml'
    )
