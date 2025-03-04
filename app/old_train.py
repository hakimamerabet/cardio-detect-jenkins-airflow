import pandas as pd
import numpy as np
import mlflow
import time
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def load_data_from_s3(bucket_name, key):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep=';')
    return df

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.dropna()
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        X = X.drop_duplicates()
        if "height" in X.columns:
            X["height"] = X["height"] / 100
        for col in ["height", "weight"]:
            if col in X.columns:
                X = X[(X[col] > X[col].quantile(0.025)) & (X[col] < X[col].quantile(0.975))]
        if "ap_hi" in X.columns and "ap_lo" in X.columns:
            X = X[(X["ap_hi"] >= 0) & (X["ap_lo"] >= 30)]
            X = X[X["ap_lo"] < X["ap_hi"]]
        if "height" in X.columns and "weight" in X.columns:
            X["bmi"] = round(X["weight"] / (X["height"] ** 2), 2)
            X = X.drop(columns=["height", "weight"])
        if "age" in X.columns and X["age"].mean() > 100:
            X["age"] = (X["age"] / 365).round()
        if "gender" in X.columns:
            X["gender"] = X["gender"].replace({1: 0, 2: 1})
        return X

def create_pipeline():
    categorical_features = ['gluc', 'cholesterol']
    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(drop='first'))
    ])
    numeric_features = ['age', 'ap_hi', 'ap_lo', 'bmi']
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return Pipeline([
        ("CustomPreprocessor", CustomPreprocessor()),
        ("ColumnTransformer", preprocessor),
        ("Random_Forest", RandomForestClassifier())
    ])

def train_model(pipe, X_train, y_train, param_grid, cv=2, n_jobs=-1, verbose=3):
    model = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    return model

def run_experiment(experiment_name, bucket_name, key, param_grid, artifact_path, registered_model_name):
    start_time = time.time()
    df = load_data_from_s3(bucket_name, key)
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = create_pipeline()
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.sklearn.autolog()
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        train_model(pipe, X_train, y_train, param_grid)
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    experiment_name = "cardio-detect"
    bucket_name = "projet-cardiodetect"
    key = "cardio_train.csv"
    param_grid = {
        "Random_Forest__max_depth": [2, 4, 6, 8, 10],
        "Random_Forest__min_samples_leaf": [1, 2, 5],
        "Random_Forest__min_samples_split": [2, 4, 8],
        "Random_Forest__n_estimators": [10, 20, 40, 60, 80, 100],
    }
    artifact_path = "modeling_cardiodetect"
    registered_model_name = "random_forest"
    run_experiment(experiment_name, bucket_name, key, param_grid, artifact_path, registered_model_name)
