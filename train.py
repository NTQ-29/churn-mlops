"""
train.py — Phoenix Checkpoint Version
-------------------------------------

This script trains a churn classification model using a full MLOps‑ready workflow:
- Loads configuration from config.yaml
- Loads and preprocesses data
- Builds a preprocessing + model pipeline
- Trains the model
- Logs metrics, parameters, and artifacts to MLflow
- Optionally registers the model in the MLflow Model Registry

This script is designed for reproducibility, traceability, and production readiness.
"""

import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)
from importlib import import_module
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file for experiment settings."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_data(train_path: str, test_path: str, target_column: str):
    """
    Load training and test datasets.
    Splits features (X) and target (y).
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# PIPELINE BUILDING
# ---------------------------------------------------------
def build_model_pipeline(categorical_features, numerical_features, model_class, model_params):
    """
    Build a full preprocessing + model pipeline.
    - Categorical features → OneHot
    """