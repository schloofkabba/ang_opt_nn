# data/mnist.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(
    dataset: str = "mnist",
    test_size: int = 10000,
    random_state: int = 42,
    dataset_limit: int | None = None,
    val_size: int = 10000,
    data_home: str | None = None,
):
    """
    Lädt MNIST (OpenML 'mnist_784') und liefert:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    mit Shapes X_* = (784, N), y_* = (N,).
    
    Parameter
    ---------
    dataset : "mnist" | "mnist_784"
        Alias; beides führt zu OpenML 'mnist_784'.
    test_size : int
        Anzahl Testbeispiele (klassisch 10_000).
    random_state : int
        Seed für reproduzierbare Splits.
    dataset_limit : int | None
        Begrenze die **Trainingsmenge** (ohne Validierung). Die Validierung
        bleibt val_size groß. Setze None für volle Datenmenge.
    val_size : int
        Anzahl Validierungsbeispiele (klassisch 10_000).
    data_home : str | None
        Optionales Cache-Verzeichnis für fetch_openml.
    """
    if dataset not in ("mnist", "mnist_784"):
        raise ValueError(f"Unsupported dataset '{dataset}'. Use 'mnist' or 'mnist_784'.")

    # 1) Laden & Vorverarbeitung
    ds = fetch_openml("mnist_784", version=1, as_frame=False, data_home=data_home)
    X = ds.data.astype(np.float32) / 255.0   # (70000, 784), 0..1
    # y kommt als str; in int umwandeln
    y = ds.target.astype(np.int64)           # (70000,)

    # 2) Testset abspalten (stratifiziert)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3) Optional: Trainings-Pool begrenzen, bevor wir die Validierung abzweigen
    if dataset_limit is not None:
        if dataset_limit <= 0:
            raise ValueError("dataset_limit must be a positive integer or None.")
        needed = dataset_limit + (val_size if val_size else 0)
        if X_pool.shape[0] < needed:
            raise ValueError(
                f"dataset_limit={dataset_limit} zu groß für verfügbaren Pool {X_pool.shape[0]} "
                f"mit val_size={val_size}. Benötigt: dataset_limit + val_size = {needed}."
            )
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_pool.shape[0], size=needed, replace=False)
        X_pool, y_pool = X_pool[idx], y_pool[idx]

    # 4) Validierung abspalten (stratifiziert)
    if val_size and val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool,
            test_size=val_size,
            random_state=random_state,
            stratify=y_pool
        )
    else:
        X_train, y_train = X_pool, y_pool
        X_val = np.empty((0, X.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=np.int64)

    # 5) In (D, N) drehen
    X_train = X_train.T  # (784, N_train)
    X_val   = X_val.T    # (784, N_val)
    X_test  = X_test.T   # (784, N_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
