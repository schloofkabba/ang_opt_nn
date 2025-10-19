from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from models.mlp import MLP




@dataclass
class GAConfig:
    pop_size: int = 20
    generations: int = 6
    sigma: float = 0.05
    elite_frac: float = 0.1
    tournament_k: int = 3
    seed: int = 42
    fitness_batch: int = 2000




def evaluate_fitness(model: MLP, genome, X, y, rng, batch):
    N = X.shape[1]
    idx = rng.choice(N, size=min(batch, N), replace=False)
    Xb = X[:, idx]
    yb = y[idx]
    model.unflatten_into(genome)
    loss, _ = model.loss_and_grads(Xb, yb)
    return -float(loss) # higher is better




def tournament_select(fitness: np.ndarray, rng, k: int) -> int:
    cand = rng.integers(0, fitness.size, size=k)
    return int(cand[np.argmax(fitness[cand])])




def ga_optimize(model: MLP, X_val, y_val, cfg: GAConfig):
    rng = np.random.default_rng(cfg.seed)
    L = model.flatten().size
    base = model.flatten()
    pop = np.stack([base + rng.normal(0, cfg.sigma, size=L).astype(np.float32)
                    for _ in range(cfg.pop_size)], axis=0)
    fitness = np.zeros(cfg.pop_size, dtype=np.float32)
    elites = max(1, int(cfg.elite_frac * cfg.pop_size))


    for gen in range(1, cfg.generations + 1):
        for i in range(cfg.pop_size):
            fitness[i] = evaluate_fitness(model, pop[i], X_val, y_val, rng, cfg.fitness_batch)
        best_idx = int(np.argmax(fitness))
        print(f"[ga] gen {gen:02d} | best_fitness={fitness[best_idx]:.5f}")


        order = np.argsort(-fitness)
        next_pop = [pop[order[e]].copy() for e in range(elites)]
        while len(next_pop) < cfg.pop_size:
            p1 = pop[tournament_select(fitness, rng, cfg.tournament_k)]
            p2 = pop[tournament_select(fitness, rng, cfg.tournament_k)]
            mask = rng.random(L) < 0.5
            child = np.where(mask, p1, p2).astype(np.float32)
            child += rng.normal(0, cfg.sigma, size=L).astype(np.float32)
            next_pop.append(child)
        pop = np.stack(next_pop, axis=0)


    for i in range(cfg.pop_size):
        fitness[i] = evaluate_fitness(model, pop[i], X_val, y_val, rng, cfg.fitness_batch)
    best_idx = int(np.argmax(fitness))
    return pop[best_idx].copy(), float(fitness[best_idx])