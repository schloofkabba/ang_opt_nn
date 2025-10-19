from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt




def show_first_layer_weights(model, max_to_show: int = 64):
    H = model.cfg.hidden_dim
    n = min(H, max_to_show)
    cols = int(math.sqrt(n)) or 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows*cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')
        if i < n:
            w = model.W1[i].reshape(28, 28)
            ax.imshow(w, cmap='gray')
    plt.suptitle('First-layer weights (as 28x28)')
    plt.tight_layout()
    plt.show()