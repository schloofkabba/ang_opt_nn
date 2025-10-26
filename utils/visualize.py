# utils/visualize.py
import math
import numpy as np
import matplotlib.pyplot as plt

def show_first_layer_weights(model, max_to_show=64):
    """
    Zeigt die ersten-Layer-Gewichte als 28x28 Tiles.
    """
    W_first = model.W[0]  # (H1, 784)
    H1, D = W_first.shape
    n = min(H1, max_to_show)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0))
    axes = np.atleast_2d(axes)

    for i in range(rows*cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis('off')
        if i < n:
            w = W_first[i, :].reshape(28, 28)
            # Skala für bessere Sichtbarkeit normalisieren
            m = np.max(np.abs(w)) + 1e-8
            ax.imshow(w, cmap='gray', vmin=-m, vmax=m)
            ax.set_title(f"h{i}", fontsize=8)
    plt.tight_layout()
    plt.show()
