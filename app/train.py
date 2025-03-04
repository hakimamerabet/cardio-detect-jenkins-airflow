import pandas as pd
import numpy as np
import mlflow
import time
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data from S3
def load_data_from_s3(bucket_name, key):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(data), sep=';')

# Custom preprocessing function
def preprocess(df):
    df = df.dropna().drop(columns=["id"], errors='ignore').drop_duplicates()
    df['height'] = df['height'] / 100
    df = df[(df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))) &
            (df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975)))]
    df = df[(df['ap_hi'] >= 0) & (df['ap_lo'] >= 30) & (df['ap_lo'] < df['ap_hi'])]
    df['bmi'] = round(df['weight'] / (df['height'] ** 2), 2)
    df = df.drop(columns=['height', 'weight'])
    df['age'] = (df['age'] / 365).round() if df['age'].mean() > 100 else df['age']
    df['gender'] = df['gender'].replace({1: 0, 2: 1})
    return df

# Define the full preprocessing pipeline
def create_pipeline():
    preprocess_transformer = FunctionTransformer(preprocess)
    categorical_features = ['gluc', 'cholesterol']
    numeric_features = ['age', 'ap_hi', 'ap_lo', 'bmi']
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return Pipeline(steps=[
        ('custom_preprocessor', preprocess_transformer),
        ('feature_processing', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

# Train model with GridSearchCV
def train_model(pipe, X_train, y_train, param_grid):
    model = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1, verbose=3, scoring="r2")
    model.fit(X_train, y_train)
    return model

# Run experiment
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
        model = train_model(pipe, X_train, y_train, param_grid)
        mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)
    print(f"...Training Done! --- Total time: {time.time() - start_time} seconds")

# Main execution
if __name__ == "__main__":
    experiment_name = "cardio-detect"
    bucket_name = "projet-cardiodetect"
    key = "cardio_train.csv"
    param_grid = {
        "classifier__max_depth": [2, 4, 6, 8, 10],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__min_samples_split": [2, 4, 8],
        "classifier__n_estimators": [10, 20, 40, 60, 80, 100],
    }
    artifact_path = "modeling_cardiodetect"
    registered_model_name = "random_forest"
    run_experiment(experiment_name, bucket_name, key, param_grid, artifact_path, registered_model_name)
