from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np




def he_init(fan_in: int, fan_out: int, rng: np.random.Generator):
    std = math.sqrt(2.0 / fan_in)
    W = rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float32)
    b = np.zeros((fan_out, 1), dtype=np.float32)
    return W, b




def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)




def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)




def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=0, keepdims=True)




def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((num_classes, y.size), dtype=np.float32)
    Y[y, np.arange(y.size)] = 1.0
    return Y




@dataclass
class NetConfig:
    input_dim: int = 784
    hidden_dim: int = 64
    output_dim: int = 10
    seed: int = 42




class MLP:
    """1-hidden-layer MLP (ReLU + Softmax) using column-major batches.
    X: (D, N) with D=784; y: (N,)
    """
    def __init__(self, cfg: NetConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.W1, self.b1 = he_init(cfg.input_dim, cfg.hidden_dim, self.rng)
        self.W2, self.b2 = he_init(cfg.hidden_dim, cfg.output_dim, self.rng)


    def forward(self, X: np.ndarray):
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        P = softmax(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P}
        return P, cache


    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[1]
        P, cache = self.forward(X)
        Y = one_hot(y, self.cfg.output_dim)
        eps = 1e-12
        loss = -np.sum(Y * np.log(P + eps)) / N
        dZ2 = (P - Y) / N
        dW2 = dZ2 @ cache["A1"].T
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * relu_grad(cache["Z1"])
        dW1 = dZ1 @ cache["X"].T
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return loss, grads


    def predict(self, X: np.ndarray) -> np.ndarray:
        P, _ = self.forward(X)
        return np.argmax(P, axis=0)


    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean(pred == y))


    def flatten(self) -> np.ndarray:
        return np.concatenate([
        self.W1.flatten(), self.b1.flatten(),
        self.W2.flatten(), self.b2.flatten()
        ]).astype(np.float32)


    def unflatten_into(self, vec: np.ndarray) -> None:
        H = self.cfg.hidden_dim
        D = self.cfg.input_dim
        C = self.cfg.output_dim
        i0 = 0
        i1 = i0 + H*D
        self.W1 = vec[i0:i1].reshape(H, D)
        i0 = i1; i1 = i0 + H
        self.b1 = vec[i0:i1].reshape(H, 1)
        i0 = i1; i1 = i0 + C*H
        self.W2 = vec[i0:i1].reshape(C, H)
        i0 = i1; i1 = i0 + C
        self.b2 = vec[i0:i1].reshape(C, 1)