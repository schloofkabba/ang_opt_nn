from __future__ import annotations
import numpy as np




class DigitGUI:
    def __init__(self, model, cell_px: int = 12):
        import tkinter as tk
        self.tk = tk.Tk()
        self.tk.title('MNIST ReLU Classifier – Draw a digit')
        self.model = model
        self.grid_size = 28
        self.cell_px = cell_px
        self.canvas_px = self.grid_size * self.cell_px
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)


        self.canvas = tk.Canvas(self.tk, width=self.canvas_px, height=self.canvas_px, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

        self.rects = []
        for r in range(self.grid_size):
            row_rects = []
            for c in range(self.grid_size):
                x0 = c * self.cell_px
                y0 = r * self.cell_px
                x1 = x0 + self.cell_px
                y1 = y0 + self.cell_px
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline='#eee', fill='white')
                row_rects.append(rect)
            self.rects.append(row_rects)


        self.canvas.bind('<B1-Motion>', self.on_paint)
        self.canvas.bind('<Button-1>', self.on_paint)


        self.tk_pred_var = tk.StringVar(value='Prediction: –')
        self.tk_topk_var = tk.StringVar(value='Top-3: –')

        pred_label = tk.Label(self.tk, textvariable=self.tk_pred_var, font=('Segoe UI', 16))
        pred_label.grid(row=0, column=1, sticky='w', padx=10)


        btn_predict = tk.Button(self.tk, text='Predict', command=self.predict_current)
        btn_predict.grid(row=1, column=1, sticky='w', padx=10, pady=(0,10))


        btn_clear = tk.Button(self.tk, text='Clear', command=self.clear)
        btn_clear.grid(row=2, column=1, sticky='w', padx=10, pady=(0,10))


        topk_label = tk.Label(self.tk, textvariable=self.tk_topk_var, font=('Consolas', 12))
        topk_label.grid(row=3, column=1, sticky='w', padx=10)


        instr = tk.Label(self.tk, text='Left-click & drag to draw. Predict to classify. Clear to reset.', fg='#555')
        instr.grid(row=4, column=1, sticky='w', padx=10)


        self.tk.bind('<space>', lambda e: self.predict_current())
        self.tk.bind('<Escape>', lambda e: self.clear())

    def run(self):
        self.tk.mainloop()


    def on_paint(self, event):
        c = int(event.x // self.cell_px)
        r = int(event.y // self.cell_px)
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            self.stamp(r, c)


    def stamp(self, r: int, c: int):
        kernel = np.array([[0.2, 0.5, 0.2],
                            [0.5, 1.0, 0.5],
                            [0.2, 0.5, 0.2]], dtype=np.float32)
        kr, kc = kernel.shape
        r0 = r - kr//2
        c0 = c - kc//2
        for i in range(kr):
            for j in range(kc):
                rr = r0 + i
                cc = c0 + j
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    self.grid[rr, cc] = np.clip(self.grid[rr, cc] + kernel[i, j], 0.0, 1.0)
                    self.update_cell(rr, cc)


    def update_cell(self, r: int, c: int):
        val = self.grid[r, c]
        g = int(255 * (1.0 - val))
        color = f'#{g:02x}{g:02x}{g:02x}'
        self.canvas.itemconfig(self.rects[r][c], fill=color)

    def clear(self):
        self.grid[:] = 0.0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.canvas.itemconfig(self.rects[r][c], fill='white')
        self.tk_pred_var.set('Prediction: –')
        self.tk_topk_var.set('Top-3: –')


    def predict_current(self):
        x = self.grid.reshape(-1, 1) # (784,1)
        probs, pred = self.forward_proba(x)
        top3 = np.argsort(-probs.flatten())[:3]
        self.tk_pred_var.set(f"Prediction: {pred}")
        self.tk_topk_var.set(f"Top-3: {[(int(k), float(probs[k])) for k in top3]}")


    def forward_proba(self, X):
        P, _ = self.model.forward(X)
        pred = int(np.argmax(P, axis=0)[0])
        return P[:, 0], pred