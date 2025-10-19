from __future__ import annotations
import os
import sys
from pathlib import Path
import yaml
from models.mlp import MLP, NetConfig
from data.mnist import load_mnist
from optim.sgd import sgd_train
from optim.ga import GAConfig, ga_optimize
from utils.io import save_model, load_model
from utils.visualize import show_first_layer_weights
from gui.draw_gui import DigitGUI

def load_config(path: str | os.PathLike) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def ensure_dirs(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

def maybe_load_or_init_model(cfg: dict):
    net_cfg = NetConfig(
        input_dim=cfg['model'].get('input_dim', 784),
        hidden_dim=cfg['model'].get('hidden_dim', 64),
        output_dim=cfg['model'].get('output_dim', 10),
        seed=cfg.get('seed', 42),
        )
    weights_cfg = cfg['weights']
    save_path = weights_cfg.get('save_path', './artifacts/mnist_relu_best.npz')
    load_if_exists = bool(weights_cfg.get('load_if_exists', True))


    if load_if_exists and os.path.exists(save_path):
        print(f"[main] Loading existing model: {save_path}")
        model = load_model(save_path)
    else:
        print("[main] Creating new model")
        model = MLP(net_cfg)
    return model

if __name__ == "__main__":
    # Allow running from VS Code without adjusting cwd
    proj_root = Path(__file__).resolve().parent
    os.chdir(proj_root)


    cfg = load_config('config.yaml')
    run_cfg = cfg['run']
    weights_cfg = cfg['weights']
    save_path = weights_cfg.get('save_path', './artifacts/mnist_relu_best.npz')


    model = maybe_load_or_init_model(cfg)


    # TRAIN
    if run_cfg.get('train', True):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
            dataset_limit=cfg['data'].get('dataset_limit'),
            test_size=cfg['data'].get('test_size', 10000),
            random_state=cfg['data'].get('random_state', 42),
        )
        trainer = cfg['train'].get('trainer', 'sgd').lower()
        print(f"[main] Trainer = {trainer}")


        if trainer == 'sgd':
            tcfg = cfg['train']['sgd']
            stats = sgd_train(
                model,
                X_train, y_train,
                X_val, y_val,
                epochs=int(tcfg.get('epochs', 5)),
                batch_size=int(tcfg.get('batch_size', 128)),
                lr=float(tcfg.get('lr', 0.05)),
                seed=int(cfg.get('seed', 42)),
                verbose=True,
            )
            print(f"[main] Best val acc: {stats['best_val_acc']:.4f}")


        elif trainer == 'ga':
            gcfg = cfg['train']['ga']
            ga_cfg = GAConfig(
                pop_size=int(gcfg.get('pop_size', 20)),
                generations=int(gcfg.get('generations', 6)),
                sigma=float(gcfg.get('sigma', 0.05)),
                elite_frac=float(gcfg.get('elite_frac', 0.1)),
                tournament_k=int(gcfg.get('tournament_k', 3)),
                seed=int(cfg.get('seed', 42)),
                fitness_batch=int(gcfg.get('fitness_batch', 2000)),
            )
            best_genome, best_fit = ga_optimize(model, X_val, y_val, ga_cfg)
            print(f"[main] GA done. Fitness: {best_fit:.5f}")
            model.unflatten_into(best_genome)


        elif trainer == 'hybrid':
            gcfg = cfg['train']['ga']
            ga_cfg = GAConfig(
                pop_size=int(gcfg.get('pop_size', 20)),
                generations=int(gcfg.get('generations', 6)),
                sigma=float(gcfg.get('sigma', 0.05)),
                elite_frac=float(gcfg.get('elite_frac', 0.1)),
                tournament_k=int(gcfg.get('tournament_k', 3)),
                seed=int(cfg.get('seed', 42)),
                fitness_batch=int(gcfg.get('fitness_batch', 2000)),
            )
            best_genome, best_fit = ga_optimize(model, X_val, y_val, ga_cfg)
            model.unflatten_into(best_genome)
            print(f"[main] Hybrid: GA init done (fitness {best_fit:.5f}). Switching to SGD...")
            tcfg = cfg['train']['sgd']
            stats = sgd_train(
                model,
                X_train, y_train,
                X_val, y_val,
                epochs=int(tcfg.get('epochs', 5)),
                batch_size=int(tcfg.get('batch_size', 128)),
                lr=float(tcfg.get('lr', 0.05)),
                seed=int(cfg.get('seed', 42)),
                verbose=True,
            )
            print(f"[main] Best val acc (post-SGD): {stats['best_val_acc']:.4f}")
        else:
            raise ValueError(f"Unknown trainer: {trainer}")


        # Evaluate & save
        test_acc = model.accuracy(X_test, y_test)
        print(f"[main] Test accuracy: {test_acc:.4f}")
        ensure_dirs(save_path)
        save_model(model, save_path)
    else:
        print("[main] Training disabled in config.yaml")


    # VIZ
    if run_cfg.get('viz', False):
        show_first_layer_weights(model, max_to_show=int(cfg['viz'].get('max_units', 64)))


    # GUI
    if run_cfg.get('gui', False):
        app = DigitGUI(model, cell_px=int(cfg['gui'].get('cell_px', 12)))
        app.run()