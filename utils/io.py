from __future__ import annotations
import os
import numpy as np
from models.mlp import MLP, NetConfig




def save_model(model: MLP, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
             input_dim=model.cfg.input_dim, hidden_dim=model.cfg.hidden_dim, output_dim=model.cfg.output_dim)
    print(f"[io] Saved model → {path}")




def load_model(path: str) -> MLP:
    data = np.load(path, allow_pickle=True)
    cfg = NetConfig(int(data["input_dim"]), int(data["hidden_dim"]), int(data["output_dim"]))
    m = MLP(cfg)
    m.W1 = data["W1"].astype(np.float32)
    m.b1 = data["b1"].astype(np.float32)
    m.W2 = data["W2"].astype(np.float32)
    m.b2 = data["b2"].astype(np.float32)
    print(f"[io] Loaded model ← {path}")
    return m