
# ============================================
# Automated Testing using Pytest
# Project: Iris Flower Classification
# ============================================

import pytest

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@pytest.fixture
def trained_model():
    """
    Fixture to load Iris dataset and train
    the Random Forest classification model.
    """

    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    return model, X_test, y_test, iris


def test_model_training(trained_model):
    """Test whether the model is trained successfully."""

    model, X_test, y_test, iris = trained_model

    assert model is not None
    assert hasattr(model, "predict")

    print("\nModel training test passed!")


def test_model_prediction(trained_model):
    """Test whether the model generates predictions."""

    model, X_test, y_test, iris = trained_model

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert len(predictions) > 0

    print("\nPrediction test passed!")


def test_model_accuracy(trained_model):
    """Test whether model accuracy is within valid limits."""

    model, X_test, y_test, iris = trained_model

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert 0 <= accuracy <= 1
    assert accuracy >= 0.80

    print("\nModel accuracy:", accuracy * 100, "%")
    print("Accuracy test passed!")


def test_valid_prediction_classes(trained_model):
    """Test whether predictions belong to valid Iris classes."""

    model, X_test, y_test, iris = trained_model

    predictions = model.predict(X_test)

    valid_classes = {0, 1, 2}

    assert set(predictions).issubset(valid_classes)

    print("\nValid prediction classes test passed!")


def test_dataset_size():
    """Test whether Iris dataset contains 150 samples."""

    iris = load_iris()

    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150

    print("\nDataset size test passed!")


def test_all_flower_classes():
    """Test whether the dataset contains three flower classes."""

    iris = load_iris()

    assert len(iris.target_names) == 3
    assert set(iris.target) == {0, 1, 2}

    print("\nFlower classes test passed!")
