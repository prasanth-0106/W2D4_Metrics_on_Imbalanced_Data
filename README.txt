# Task 1 - Metrics on Imbalanced Data

## Files
- `task1_imbalanced_metrics.py` - complete solution for Task 1.

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python task1_imbalanced_metrics.py
```

The script creates an imbalanced dataset using `weights=[0.95, 0.05]`, evaluates a
majority-class baseline, Logistic Regression, and Logistic Regression with
`class_weight="balanced"`. It prints accuracy, precision, recall, F1, a
classification report, a confusion matrix, and fraud-example explanations of TP,
FP, FN, and TN.

The exact numeric metrics are produced by the script for the installed
scikit-learn version. The assignment's majority-class example is approximately
0.95 accuracy, 0.00 recall, and 0.00 F1.
