# models/mlp.py
from dataclasses import dataclass
import numpy as np

def he_init(shape, rng):
    fan_in = shape[1]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape).astype(np.float32)

@dataclass
class NetConfig:
    input_dim: int
    output_dim: int
    hidden_dims: list  # [H] oder [H1, H2]

class MLP:
    def __init__(self, cfg: NetConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        D = int(cfg.input_dim)
        C = int(cfg.output_dim)
        Hs = list(cfg.hidden_dims)
        assert len(Hs) in (1, 2), "Only 1 or 2 hidden layers are supported."

        self.num_hidden = len(Hs)
        self.W = []
        self.b = []

        # Layer 1: D -> H1
        H1 = int(Hs[0])
        self.W.append(he_init((H1, D), self.rng))
        self.b.append(np.zeros((H1, 1), dtype=np.float32))

        if self.num_hidden == 2:
            # Layer 2: H1 -> H2
            H2 = int(Hs[1])
            self.W.append(he_init((H2, H1), self.rng))
            self.b.append(np.zeros((H2, 1), dtype=np.float32))
            # Output: H2 -> C
            self.W.append(he_init((C, H2), self.rng))
            self.b.append(np.zeros((C, 1), dtype=np.float32))
        else:
            # Output: H1 -> C
            self.W.append(he_init((C, H1), self.rng))
            self.b.append(np.zeros((C, 1), dtype=np.float32))

    # ---------- Convenience (Backwards-Kompatibilität) ----------
    @property
    def W1(self): return self.W[0]
    @property
    def b1(self): return self.b[0]
    @property
    def W2(self): return self.W[1] if self.num_hidden == 1 else self.W[2]
    @property
    def b2(self): return self.b[1] if self.num_hidden == 1 else self.b[2]
    @property
    def W_hidden2(self): return None if self.num_hidden == 1 else self.W[1]
    @property
    def b_hidden2(self): return None if self.num_hidden == 1 else self.b[1]

    # ---------- Forward ----------
    def forward(self, X):
        """
        X: (D, N)
        returns P: (C, N), cache: intermediates for backprop
        """
        if self.num_hidden == 1:
            Z1 = self.W[0] @ X + self.b[0]                   # (H1,N)
            A1 = np.maximum(0, Z1)                           # ReLU
            Z2 = self.W[1] @ A1 + self.b[1]                  # (C,N)
            Z2m = Z2 - Z2.max(axis=0, keepdims=True)         # Stable Softmax
            expZ = np.exp(Z2m)
            P = expZ / expZ.sum(axis=0, keepdims=True)
            cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
            return P, cache
        else:
            Z1 = self.W[0] @ X + self.b[0]                   # (H1,N)
            A1 = np.maximum(0, Z1)
            Z2 = self.W[1] @ A1 + self.b[1]                  # (H2,N)
            A2 = np.maximum(0, Z2)
            Z3 = self.W[2] @ A2 + self.b[2]                  # (C,N)
            Z3m = Z3 - Z3.max(axis=0, keepdims=True)
            expZ = np.exp(Z3m)
            P = expZ / expZ.sum(axis=0, keepdims=True)
            cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3}
            return P, cache

    # ---------- Loss & Gradients ----------
    def loss_and_grads(self, X, y):
        """
        y: (N,) int labels 0..C-1
        returns (loss, grads_dict)
        """
        P, cache = self.forward(X)
        N = X.shape[1]
        logp = -np.log(P[y, range(N)] + 1e-12)
        loss = float(np.mean(logp))

        dZout = P.copy()
        dZout[y, range(N)] -= 1.0
        dZout /= N

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        if self.num_hidden == 1:
            A1 = cache["A1"]
            dW2 = dZout @ A1.T
            db2 = dZout.mean(axis=1, keepdims=True)

            dA1 = self.W[1].T @ dZout
            dZ1 = dA1 * (cache["Z1"] > 0)
            dW1 = dZ1 @ cache["X"].T
            db1 = dZ1.mean(axis=1, keepdims=True)

            grads_W[1], grads_b[1] = dW2.astype(np.float32), db2.astype(np.float32)
            grads_W[0], grads_b[0] = dW1.astype(np.float32), db1.astype(np.float32)
        else:
            A2 = cache["A2"]; A1 = cache["A1"]
            dW3 = dZout @ A2.T
            db3 = dZout.mean(axis=1, keepdims=True)

            dA2 = self.W[2].T @ dZout
            dZ2 = dA2 * (cache["Z2"] > 0)
            dW2 = dZ2 @ A1.T
            db2 = dZ2.mean(axis=1, keepdims=True)

            dA1 = self.W[1].T @ dZ2
            dZ1 = dA1 * (cache["Z1"] > 0)
            dW1 = dZ1 @ cache["X"].T
            db1 = dZ1.mean(axis=1, keepdims=True)

            grads_W[2], grads_b[2] = dW3.astype(np.float32), db3.astype(np.float32)
            grads_W[1], grads_b[1] = dW2.astype(np.float32), db2.astype(np.float32)
            grads_W[0], grads_b[0] = dW1.astype(np.float32), db1.astype(np.float32)

        grads = {"W": grads_W, "b": grads_b}
        return loss, grads

    # ---------- Accuracy ----------
    def accuracy(self, X, y):
        P, _ = self.forward(X)
        pred = np.argmax(P, axis=0)
        return float((pred == y).mean())

    # ---------- Flatten/Unflatten ----------
    def flatten(self):
        vecs = []
        for w in self.W: vecs.append(w.ravel())
        for b in self.b: vecs.append(b.ravel())
        return np.concatenate(vecs).astype(np.float32)

    def unflatten_into(self, vec: np.ndarray):
        vec = vec.astype(np.float32, copy=False)
        off = 0

        # Gewichte
        for i, w in enumerate(self.W):
            n = w.size
            self.W[i][...] = vec[off:off+n].reshape(w.shape)
            off += n
        # Bias
        for i, b in enumerate(self.b):
            n = b.shape[0]
            self.b[i][..., 0] = vec[off:off+n]
            off += n

        assert off == vec.size, "Vector size mismatch in unflatten."
