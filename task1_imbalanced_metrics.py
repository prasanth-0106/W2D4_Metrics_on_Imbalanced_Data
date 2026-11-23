"""
Task 1 - Metrics on Imbalanced Data
Run: python task1_imbalanced_metrics.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def print_metrics(model_name, y_true, y_pred):
    print(f"\n{model_name}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 score : {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main():
    
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)
    majority_pred = majority_model.predict(X_test)
    print_metrics("Majority-class model", y_test, majority_pred)

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    logistic_model.fit(X_train, y_train)
    logistic_pred = logistic_model.predict(X_test)

    print("\nLogistic Regression - classification report")
    print(classification_report(y_test, logistic_pred, zero_division=0))

    cm = confusion_matrix(y_test, logistic_pred, labels=[0, 1])
    print("Confusion matrix (rows = actual [0, 1], columns = predicted [0, 1]):")
    print(cm)
    print_metrics("Logistic Regression", y_test, logistic_pred)

    balanced_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    balanced_model.fit(X_train, y_train)
    balanced_pred = balanced_model.predict(X_test)
    print_metrics("Logistic Regression (class_weight='balanced')", y_test, balanced_pred)

    print("\nConfusion matrix explanation using a fraud example:")
    print("TP (True Positive): A fraudulent transaction correctly flagged as fraud.")
    print("FP (False Positive): A genuine transaction incorrectly flagged as fraud.")
    print("FN (False Negative): A fraudulent transaction incorrectly allowed as genuine.")
    print("TN (True Negative): A genuine transaction correctly identified as genuine.")


if __name__ == "__main__":
    main()
