# main.py
import os
import yaml
import numpy as np

from models.mlp import MLP, NetConfig
from utils.io import save_model, load_model, ensure_dir
from utils.visualize import show_first_layer_weights

# externe Module aus deinem Projekt
from data.mnist import load_mnist
from optim.sgd import sgd_train
from optim.ga import GAConfig, ga_optimize  # <== deine bestehende GA-Implementierung

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Rückwärtskompat: hidden_dim -> hidden_dims
    mcfg = cfg.get("model", {})
    if "hidden_dims" not in mcfg:
        if "hidden_dim" in mcfg:
            mcfg["hidden_dims"] = [int(mcfg["hidden_dim"])]
        else:
            mcfg["hidden_dims"] = [64]
    cfg["model"] = mcfg
    return cfg

def main():
    cfg = load_config("config.yaml")
    rng = np.random.default_rng(cfg.get("seed", 42))

    # Modell aufbauen
    net_cfg = NetConfig(
        input_dim=int(cfg["model"]["input_dim"]),
        output_dim=int(cfg["model"]["output_dim"]),
        hidden_dims=[int(h) for h in cfg["model"]["hidden_dims"]],
    )
    model = MLP(net_cfg, seed=cfg.get("seed", 42))

    save_path = cfg["weights"]["save_path"]
    do_train  = bool(cfg["run"]["train"])
    do_gui    = bool(cfg["run"]["gui"])
    do_viz    = bool(cfg["run"]["viz"])

    loaded = False
    if cfg["weights"]["load_if_exists"] and os.path.exists(save_path):
        model = load_model(save_path, seed=cfg.get("seed", 42))
        loaded = True

    if do_train and not loaded:
        # Daten laden
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
            dataset=cfg["data"]["dataset"],
            test_size=int(cfg["data"]["test_size"]),
            random_state=int(cfg["data"]["random_state"]),
            dataset_limit=cfg["data"]["dataset_limit"],
        )

        trainer = cfg["train"]["trainer"].lower()
        if trainer == "sgd":
            sgd_cfg = cfg["train"]["sgd"]
            best_val = sgd_train(
                model,
                X_train, y_train,
                X_val, y_val,
                epochs=int(sgd_cfg["epochs"]),
                batch_size=int(sgd_cfg["batch_size"]),
                lr=float(sgd_cfg["lr"]),
                seed=cfg.get("seed", 42),
            )
            print(f"[main] SGD best val acc = {best_val:.4f}")

        elif trainer == "ga":
            ga_cfg = cfg["train"]["ga"]
            gac = GAConfig(
                pop_size=int(ga_cfg["pop_size"]),
                generations=int(ga_cfg["generations"]),
                sigma=float(ga_cfg["sigma"]),
                elite_frac=float(ga_cfg["elite_frac"]),
                tournament_k=int(ga_cfg["tournament_k"]),
                seed=int(cfg.get("seed", 42)),
                fitness_batch=int(ga_cfg["fitness_batch"]),
            )
            best_vec, best_fit = ga_optimize(model, X_val, y_val, gac)
            model.unflatten_into(best_vec)
            print(f"[main] GA best fitness = {best_fit:.6f}")

        elif trainer == "hybrid":
            # Beispiel: GA -> SGD
            ga_cfg = cfg["train"]["ga"]
            gac = GAConfig(
                pop_size=int(ga_cfg["pop_size"]),
                generations=int(ga_cfg["generations"]),
                sigma=float(ga_cfg["sigma"]),
                elite_frac=float(ga_cfg["elite_frac"]),
                tournament_k=int(ga_cfg["tournament_k"]),
                seed=int(cfg.get("seed", 42)),
                fitness_batch=int(ga_cfg["fitness_batch"]),
            )
            best_vec, best_fit = ga_optimize(model, X_val, y_val, gac)
            model.unflatten_into(best_vec)
            print(f"[main] GA best fitness = {best_fit:.6f}")

            sgd_cfg = cfg["train"]["sgd"]
            best_val = sgd_train(
                model,
                X_train, y_train,
                X_val, y_val,
                epochs=int(sgd_cfg["epochs"]),
                batch_size=int(sgd_cfg["batch_size"]),
                lr=float(sgd_cfg["lr"]),
                seed=cfg.get("seed", 42),
            )
            print(f"[main] Hybrid (after SGD) best val acc = {best_val:.4f}")

        else:
            raise ValueError(f"Unknown trainer: {trainer}")

        # Nach Training speichern
        ensure_dir(save_path)
        save_model(model, save_path)

        # Test-Performance (optional)
        test_acc = model.accuracy(X_test, y_test)
        print(f"[main] Test accuracy = {test_acc:.4f}")

    # Visualisierung
    if do_viz:
        show_first_layer_weights(model, max_to_show=int(cfg["viz"]["max_units"]))

    # GUI
    if do_gui:
        from gui.draw_gui import DigitGUI
        app = DigitGUI(model, cell_px=int(cfg["gui"]["cell_px"]))
        app.run()

if __name__ == "__main__":
    main()
