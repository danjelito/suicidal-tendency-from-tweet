from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from collections import namedtuple


def split_dataset(x, y, test_size=0.2, val_size=0.1, random_state=None):
    """Split the dataset into train, validation, and test sets."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Adjust validation size to the remaining training set
    val_fraction_of_train = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train,
        random_state=random_state,
    )
    # Ensure all subsets are numpy arrays
    subsets = tuple(
        subset.values if isinstance(subset, pd.Series) else subset
        for subset in (x_train, x_val, x_test, y_train, y_val, y_test)
    )
    return subsets


def return_clf_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0.0)
    recall = recall_score(y_true, y_pred, average="weighted")
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    Scores = namedtuple("Scores", ["acc", "f1", "precision", "recall", "matrix"])
    return Scores(acc, f1, precision, recall, matrix)
