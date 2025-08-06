def train_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    data = pd.read_csv('data/cleaned_telco.csv')
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, 'model/churn_model.pkl')
