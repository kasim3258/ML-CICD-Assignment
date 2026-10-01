
# ============================================
# CI/CD Pipeline for a Machine Learning Model
# Project: Iris Flower Classification
# Algorithm: Random Forest Classifier
# ============================================

import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    """
    Load Iris dataset, train Random Forest model,
    make predictions and calculate accuracy.
    """

    # Step 1: Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Step 2: Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 3: Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Step 4: Train the model
    model.fit(X_train, y_train)

    # Step 5: Make predictions
    predictions = model.predict(X_test)

    # Step 6: Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Step 7: Save model results
    results = {
        "model_name": "Random Forest Classifier",
        "dataset": "Iris Dataset",
        "accuracy": round(float(accuracy) * 100, 2),
        "total_samples": len(X),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "classes": iris.target_names.tolist()
    }

    Path("results.json").write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    # Step 8: Display results
    print("\n===================================")
    print(" IRIS FLOWER CLASSIFICATION MODEL")
    print("===================================")

    print("Model:", results["model_name"])
    print("Dataset:", results["dataset"])
    print("Total Samples:", results["total_samples"])
    print("Training Samples:", results["training_samples"])
    print("Testing Samples:", results["testing_samples"])

    print("\nModel Accuracy:", results["accuracy"], "%")

    print("\nSample Predictions:")

    for i in range(min(5, len(predictions))):
        actual = iris.target_names[y_test[i]]
        predicted = iris.target_names[predictions[i]]

        print("Actual:", actual, "| Predicted:", predicted)

    print("\nModel training completed successfully!")

    return model, accuracy


# Run the model
if __name__ == "__main__":
    train_model()
