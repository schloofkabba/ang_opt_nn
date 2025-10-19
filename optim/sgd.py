from __future__ import annotations
import numpy as np
from models.mlp import MLP




def sgd_train(model: MLP, X, y, X_val, y_val, *, epochs=5, batch_size=128, lr=0.05, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    indices = np.arange(N)
    best_val = 0.0
    best_state = model.flatten().copy()


    for ep in range(1, epochs + 1):
        rng.shuffle(indices)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = indices[start:end]
            Xb = X[:, idx]
            yb = y[idx]
            loss, grads = model.loss_and_grads(Xb, yb)
            model.W1 -= lr * grads['dW1']
            model.b1 -= lr * grads['db1']
            model.W2 -= lr * grads['dW2']
            model.b2 -= lr * grads['db2']
        tr = model.accuracy(X, y)
        val = model.accuracy(X_val, y_val)
        if verbose:
            print(f"[sgd] epoch {ep:02d} | train_acc={tr:.4f} | val_acc={val:.4f}")
        if val > best_val:
            best_val = val
            best_state = model.flatten().copy()
    model.unflatten_into(best_state)
    return {"best_val_acc": best_val}