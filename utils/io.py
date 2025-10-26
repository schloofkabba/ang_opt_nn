# utils/io.py
import os
import numpy as np
from models.mlp import MLP, NetConfig

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_model(model: MLP, path: str):
    ensure_dir(path)
    payload = {
        "input_dim": np.int32(model.cfg.input_dim),
        "output_dim": np.int32(model.cfg.output_dim),
        "hidden_dims": np.array(model.cfg.hidden_dims, dtype=np.int32),
    }
    for i, w in enumerate(model.W):
        payload[f"W{i+1}"] = w.astype(np.float32)
    for i, b in enumerate(model.b):
        payload[f"b{i+1}"] = b.astype(np.float32)
    np.savez_compressed(path, **payload)
    print(f"[io] saved weights -> {path}")

def load_model(path: str, seed: int = 42) -> MLP:
    d = np.load(path, allow_pickle=True)
    D = int(d["input_dim"])
    C = int(d["output_dim"])
    if "hidden_dims" in d.files:
        Hs = [int(x) for x in d["hidden_dims"].tolist()]
    else:
        # Rückwärtskompatibilität (falls alte Dateien ohne hidden_dims)
        W1 = d["W1"]; Hs = [int(W1.shape[0])]
        if "W3" in d.files:
            W2 = d["W2"]; Hs.append(int(W2.shape[0]))
    cfg = NetConfig(input_dim=D, output_dim=C, hidden_dims=Hs)
    model = MLP(cfg, seed=seed)

    for i in range(len(model.W)):
        model.W[i][...] = d[f"W{i+1}"].astype(np.float32)
    for i in range(len(model.b)):
        model.b[i][...] = d[f"b{i+1}"].astype(np.float32)

    print(f"[io] loaded weights <- {path} (hidden_dims={Hs})")
    return model
