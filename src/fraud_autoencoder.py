
"""
fraud_autoencoder.py

Build and evaluate a fraud detection model using
PyOD's AutoEncoder on the Kaggle credit card fraud dataset.

Usage:
    python -m src.fraud_autoencoder
or:
    python src/fraud_autoencoder.py

Author: Unique Karanjit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from pyod.models.auto_encoder import AutoEncoder


def load_data() -> pd.DataFrame:
    """
    Load the credit card transactions dataset from the local data folder.

    Expects the file structure:
        project_root/
            data/
                creditcard.csv

    Returns
    -------
    df : pd.DataFrame
        Loaded dataset.
    """
    # Resolve path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(current_dir, "..", "data", "creditcard.csv"))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Split the data into features and labels, then scale features.

    The Kaggle dataset has:
    - PCA features: V1..V28
    - 'Amount'
    - 'Class' (0 = normal, 1 = fraud)

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Separate features (X) and labels (y)
    if "Class" not in df.columns:
        raise KeyError("Expected a 'Class' column in the dataset indicating fraud labels.")

    X = df.drop("Class", axis=1)
    y = df["Class"].values

    # Train/test split: keep the original fraud/non-fraud distribution (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize numerical features (most are already PCA components,
    # but 'Amount' is not scaled)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_autoencoder(X_train: np.ndarray) -> AutoEncoder:
    """
    Train a PyOD AutoEncoder model.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.

    Returns
    -------
    model : AutoEncoder
        Trained AutoEncoder model.
    """
    # AutoEncoder hyperparameters can be tuned. Here we use a simple configuration.
    model = AutoEncoder(
        contamination=0.001,          # good for credit card fraud dataset
        preprocessing=True,
        lr=0.001,
        epoch_num=20,                 # replaces epochs
        batch_size=256,
        optimizer_name='adam',
        random_state=42,
        verbose=1,
        hidden_neuron_list=[32, 16],  # bottleneck architecture
        hidden_activation_name='relu',
        batch_norm=True,
        dropout_rate=0.1
    )

    model.fit(X_train)
    return model


def evaluate_model(model: AutoEncoder, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the AutoEncoder on test set and print metrics.

    PyOD uses:
    - model.labels_: binary labels for training data
    - model.decision_scores_: raw outlier scores for training data
    - model.predict(X): binary labels for new data (0 = inlier, 1 = outlier)
    - model.decision_function(X): raw outlier scores for new data

    For fraud detection, we interpret "outlier" = fraud.
    """
    # PyOD predicted labels on test set (0: normal, 1: anomaly)
    y_pred = model.predict(X_test)

    # Raw anomaly scores (higher = more anomalous)
    y_scores = model.decision_function(X_test)

    print("=== Confusion Matrix (0: normal, 1: fraud) ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_scores)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot ROC curve (you can use this plot for a screenshot)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AutoEncoder (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AutoEncoder Fraud Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print("Class distribution (0 = normal, 1 = fraud):")
    print(df["Class"].value_counts())

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\nTraining AutoEncoder model...")
    model = train_autoencoder(X_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
