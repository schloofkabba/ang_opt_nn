from __future__ import annotations
import numpy as np




def load_mnist(dataset_limit: int | None = None, test_size: int = 10000, random_state: int = 42):
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
    except Exception:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")


    print("[data] Loading MNIST (OpenML: mnist_784)…")
    ds = fetch_openml('mnist_784', version=1, as_frame=False)
    X = ds.data.astype(np.float32) / 255.0
    y = ds.target.astype(np.int64)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


    if dataset_limit is not None and dataset_limit < X_train.shape[0]:
        X_train = X_train[:dataset_limit]
        y_train = y_train[:dataset_limit]


    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=10000, random_state=random_state, stratify=y_train
    )


    return (X_train.T, y_train), (X_val.T, y_val), (X_test.T, y_test)