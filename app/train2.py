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
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from typing import Any

# Define a custom PythonModel class to wrap the sklearn pipeline
class CardioDetectionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.Series:
        """
        Predict method to make predictions using the pipeline.

        Parameters:
        - context: Information about the prediction context.
        - model_input: The input data in DataFrame format.

        Returns:
        - The predicted values as a pandas Series.
        """
        return self.pipeline.predict(model_input)

# Load data from S3
def load_data_from_s3(bucket_name, file_key):
    """
    Load dataset from the given S3 bucket and file key.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the S3 bucket.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print(f"Loading data from S3 bucket: {bucket_name}, file key: {file_key}")
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(csv_content))

# Preprocess data
def preprocess_data(df):
    """
    Split the dataframe into X (features) and y (target).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple: Split data (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the preprocessing pipeline
def create_preprocessing_pipeline():
    # Categorial variables pipeline
    categorical_features = ['gluc', 'cholesterol']
    categorical_transformer = Pipeline(
        steps=[
        ('encoder', OneHotEncoder(drop='first')) # on encode les catégories sous forme de colonnes comportant des 0 et des 1
    ])

    # Quantitative variables pipeline
    numeric_features = ['age', 'ap_hi', 'ap_lo', 'bmi']
    numeric_transformer = Pipeline(
        steps=[
        ('scaler', StandardScaler()) # pour normaliser les variables
    ])

    # Pipelines combination
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

# Create the full pipeline
def create_pipeline():
    """
    Create a machine learning pipeline with StandardScaler and RandomForestRegressor.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    preprocessor = create_preprocessing_pipeline()
    return Pipeline(steps=[
        ("Preprocessing", preprocessor),
        ("Random_Forest", RandomForestClassifier())
    ])

# Train model
def train_model(pipe, X_train, y_train, param_grid, cv=2, n_jobs=-1, verbose=3):
    """
    Train the model using GridSearchCV.

    Args:
        pipe (Pipeline): The pipeline to use for training.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        param_grid (dict): The hyperparameter grid to search over.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of jobs to run in parallel.
        verbose (int): Verbosity level.

    Returns:
        GridSearchCV: Trained GridSearchCV object.
    """
    model = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

# Log metrics and model to MLflow
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    """
    Log training and test metrics, and the model to MLflow.

    Args:
        model (GridSearchCV): The trained model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))

    # Log the entire pipeline model (preprocessing + model) using custom PythonModel class
    cardio_model = CardioDetectionModel(model.best_estimator_)

    mlflow.pyfunc.log_model(
        artifact_path,  # Model name
        python_model=cardio_model,  # Model class instance
        signature=infer_signature(X_train, model.best_estimator_.predict(X_train)),
        input_example=X_train.iloc[:1]
    )
    print(f"Model logged to MLflow under artifact path: {artifact_path}")

# Main function to execute the workflow
def run_experiment(experiment_name, bucket_name, file_key, param_grid, artifact_path, registered_model_name):
    """
    Run the entire ML experiment pipeline.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the S3 bucket.
        param_grid (dict): The hyperparameter grid for GridSearchCV.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    # Start timing
    start_time = time.time()

    # Load and preprocess data
    df = load_data_from_s3(bucket_name, file_key)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create pipeline
    pipe = create_pipeline()

    # Set experiment's info
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        model = train_model(pipe, X_train, y_train, param_grid)

        # Log metrics and model
        log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

# Entry point for the script
if __name__ == "__main__":
    # Define experiment parameters
    experiment_name = "cardio-detect"
    bucket_name = "projet-cardiodetect"
    file_key = "cardio_train.csv"

    param_grid = {
        "Random_Forest__max_depth": [2, 4, 6, 8, 10],
        "Random_Forest__min_samples_leaf": [1, 2, 5],
        "Random_Forest__min_samples_split": [2, 4, 8],
        "Random_Forest__n_estimators": [10, 20, 40, 60, 80, 100],
    }
    artifact_path = "modeling_cardiodetect"
    registered_model_name = "random_forest"

    # Run the experiment
    run_experiment(experiment_name, bucket_name, file_key, param_grid, artifact_path, registered_model_name)
