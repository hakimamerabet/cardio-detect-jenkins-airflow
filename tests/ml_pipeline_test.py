import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline, train_model

# S3 bucker_name & key to be use for all functions
bucket_name = "hk-cardio-detect-project"
key = "cardio_train.csv"

# Test data loading
def test_load_data_from_s3():
    # Load data from S3
    df = load_data_from_s3(bucket_name, key)
    # Assertions to validate data loading
    assert not df.empty, "Dataframe is empty"
    assert 'cardio' in df.columns, "Target column is missing"

# Test data preprocessing
def test_preprocess_data():
    # Load data from S3
    df = load_data_from_s3(bucket_name, key)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Assertions to validate preprocessing
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"
    assert 'bmi' in X_train.columns, "BMI column is missing"
    assert X_train['gender'].nunique() == 2, "Gender modification failed"


# Test pipeline creation
def test_create_pipeline():
    pipe = create_pipeline()
    assert "standard_scaler" in pipe.named_steps, "Scaler missing in pipeline"
    assert "Random_Forest" in pipe.named_steps, "RandomForest missing in pipeline"

# Test model training
@mock.patch('app.train.GridSearchCV')
def test_train_model(mock_grid_search):
    # Mock GridSearchCV to avoid actual model training
    mock_grid_search.return_value.fit.return_value = mock_grid_search

    # Initialize the data pipeline
    data_pipeline = DataPipeline(bucket_name, key)

    # Load and preprocess data from S3
    df = data_pipeline.load_data()
    X_train, X_test, y_train, y_test = data_pipeline.preprocess_data(df)

    # Create a pipeline
    pipe = data_pipeline.create_pipeline()

    # Define a simple param_grid for testing
    param_grid = {
        "Random_Forest__n_estimators": [10],
        "Random_Forest__max_depth": [2]
    }

    # Train the model
    model = train_model(pipe, X_train, y_train, param_grid)

    # Assertions to validate the training
    assert model is not None, "Model training failed"
    assert hasattr(model, 'best_score_'), "Model does not have a best score"
    assert model.best_score_ > 0, "Model's best score is not reasonable"
