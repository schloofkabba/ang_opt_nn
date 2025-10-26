# optim/sgd.py
import numpy as np

def sgd_train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    N = X_train.shape[1]

    best_val = -np.inf
    best_vec = model.flatten().copy()

    for epoch in range(1, epochs + 1):
        # Shuffle
        idx = rng.permutation(N)
        X_tr = X_train[:, idx]
        y_tr = y_train[idx]

        # Mini-batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_tr[:, start:end]
            yb = y_tr[start:end]

            loss, grads = model.loss_and_grads(Xb, yb)

            # Parameter-Update (listenbasiert)
            for i in range(len(model.W)):
                model.W[i] -= lr * grads["W"][i]
            for i in range(len(model.b)):
                model.b[i] -= lr * grads["b"][i]

        # Logging pro Epoche
        train_acc = model.accuracy(X_train, y_train)
        val_acc   = model.accuracy(X_val, y_val)
        print(f"[sgd] epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # Best-State sichern
        if val_acc > best_val:
            best_val = val_acc
            best_vec = model.flatten().copy()

    # bestes Modell zurückspielen
    model.unflatten_into(best_vec)
    return best_val
