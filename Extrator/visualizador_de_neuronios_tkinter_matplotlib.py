
import json
import math
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * t

def val_to_hex_color_diverging(v, vmin, vmax):
    if vmax <= 0 and vmin >= 0:
        vmax = 1.0
        vmin = -1.0
    maxabs = max(abs(vmin), abs(vmax), 1e-8)
    x = max(-1.0, min(1.0, v / maxabs))
    if x >= 0:
        r = int(lerp(255, 255, x))
        g = int(lerp(255, 64,  x))
        b = int(lerp(255, 64,  x))
    else:
        x = -x
        r = int(lerp(64,  255, x))
        g = int(lerp(64,  255, x))
        b = int(lerp(255, 255, x))
    return f"#{r:02x}{g:02x}{b:02x}"

def val_to_hex_color_sequential(v, vmin, vmax):
    if vmax - vmin <= 1e-12:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    r = int(lerp(255, 50,  t))
    g = int(lerp(255, 200, t))
    b = int(lerp(255, 50,  t))
    return f"#{r:02x}{g:02x}{b:02x}"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_dense_layers_dict(blob):
    try:
        return blob["dense"]["layers"]
    except Exception:
        return None

def get_layer_out_neuron_count(layer_info):
    ra = layer_info.get("ref_acts", [])
    if isinstance(ra, list) and len(ra) > 0 and isinstance(ra[0], list):
        return len(ra[0])
    return int(layer_info.get("out_features", 0))

def pick_relevance_vector_for_layer(layer_name, layers_dict, prefer_self=True):
    names = list(layers_dict.keys())
    cur_idx = names.index(layer_name)
    target_count = get_layer_out_neuron_count(layers_dict[layer_name])

    if prefer_self:
        hm = layers_dict[layer_name].get("heatmap", None)
        if isinstance(hm, list) and len(hm) == target_count:
            return hm, layer_name

    for j in range(cur_idx + 1, len(names)):
        cand = layers_dict[names[j]].get("heatmap", None)
        if isinstance(cand, list) and len(cand) == target_count:
            return cand, names[j]

    hm = layers_dict[layer_name].get("heatmap", None)
    if isinstance(hm, list) and len(hm) > 0:
        return hm, layer_name

    return None, None

class DenseViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dense Layer Viewer — Grad×Act & Activations")
        self.geometry("1060x760")
        self.minsize(900, 620)

        self.json_path = None
        self.blob = None
        self.layers_dict = None

        self.current_layer_name = None
        self.mode_var = tk.StringVar(value="activation")
        self.xref_index_var = tk.IntVar(value=0)
        self.threshold_var = tk.DoubleVar(value=0.0)
        self.circle_size_var = tk.IntVar(value=22)
        self.padding_var = tk.IntVar(value=6)
        self.relu_clamp_var = tk.BooleanVar(value=True)  # NEW: clamp negatives in activation view

        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.btn_open = ttk.Button(top, text="Abrir JSON…", command=self.on_open_json); self.btn_open.pack(side=tk.LEFT)

        ttk.Label(top, text=" Layer: ").pack(side=tk.LEFT)
        self.layer_combo = ttk.Combobox(top, state="readonly", width=22)
        self.layer_combo.bind("<<ComboboxSelected>>", self.on_layer_changed)
        self.layer_combo.pack(side=tk.LEFT, padx=(0, 10))

        self.rb_act = ttk.Radiobutton(top, text="Ativação (x_ref)", variable=self.mode_var, value="activation", command=self.redraw)
        self.rb_rel = ttk.Radiobutton(top, text="Relevância (Grad×Act)", variable=self.mode_var, value="relevance", command=self.redraw)
        self.rb_act.pack(side=tk.LEFT); self.rb_rel.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(top, text=" x_ref idx: ").pack(side=tk.LEFT, padx=(10, 0))
        self.spin_xref = tk.Spinbox(top, from_=0, to=0, width=6, textvariable=self.xref_index_var, command=self.redraw)
        self.spin_xref.pack(side=tk.LEFT)

        ttk.Label(top, text=" Limiar |v|: ").pack(side=tk.LEFT, padx=(10, 0))
        self.entry_thr = tk.Entry(top, width=7, textvariable=self.threshold_var); self.entry_thr.pack(side=tk.LEFT)

        ttk.Label(top, text=" Diâmetro: ").pack(side=tk.LEFT, padx=(10, 0))
        self.entry_sz = tk.Entry(top, width=5, textvariable=self.circle_size_var); self.entry_sz.pack(side=tk.LEFT)

        ttk.Label(top, text=" Espaço: ").pack(side=tk.LEFT, padx=(10, 0))
        self.entry_pad = tk.Entry(top, width=5, textvariable=self.padding_var); self.entry_pad.pack(side=tk.LEFT)

        # NEW: ReLU clamp toggle
        self.cb_relu = ttk.Checkbutton(top, text="Aplicar ReLU (ativação)", variable=self.relu_clamp_var, command=self.redraw)
        self.cb_relu.pack(side=tk.LEFT, padx=(12, 0))

        self.btn_redraw = ttk.Button(top, text="Atualizar", command=self.redraw); self.btn_redraw.pack(side=tk.RIGHT)

        self.info_lbl = ttk.Label(self, text="Abra um JSON para começar.", anchor="w")
        self.info_lbl.pack(side=tk.TOP, fill=tk.X, padx=10)

        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.status = ttk.Label(self, text="", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def on_open_json(self):
        fp = filedialog.askopenfilename(
            title="Selecione o arquivo JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            blob = load_json(fp)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao abrir JSON:\n{e}")
            return

        layers = get_dense_layers_dict(blob)
        if not layers or not isinstance(layers, dict):
            messagebox.showerror("Erro", "JSON não contém 'dense.layers' no formato esperado.")
            return

        self.json_path = fp
        self.blob = blob
        self.layers_dict = layers

        names = list(layers.keys())
        self.layer_combo["values"] = names
        if names:
            self.layer_combo.current(0)
            self.current_layer_name = names[0]

        any_layer = layers[names[0]]
        ref_acts = any_layer.get("ref_acts", [])
        n_refs = len(ref_acts) if isinstance(ref_acts, list) else 0
        self.spin_xref.config(to=max(0, n_refs - 1))
        self.xref_index_var.set(0)

        self.info_lbl.config(text=f"Arquivo: {os.path.basename(fp)}  |  Camadas densas: {', '.join(names)}")
        self.redraw()

    def on_layer_changed(self, _evt):
        self.current_layer_name = self.layer_combo.get()
        self.redraw()

    def get_values_for_current(self):
        if not self.layers_dict or self.current_layer_name is None:
            return [], "Nenhuma camada carregada.", (0.0, 1.0)

        layer = self.layers_dict[self.current_layer_name]
        mode = self.mode_var.get()

        if mode == "activation":
            xidx = int(self.xref_index_var.get())
            acts = layer.get("ref_acts", [])
            if not isinstance(acts, list) or len(acts) == 0:
                return [], "Camada sem 'ref_acts'.", (0.0, 1.0)
            if xidx < 0 or xidx >= len(acts):
                xidx = 0
                self.xref_index_var.set(0)
            vec = list(acts[xidx])
            # Optional: emulate post-ReLU
            if self.relu_clamp_var.get():
                vec = [max(0.0, float(v)) for v in vec]
            vmin = min(vec) if len(vec) else 0.0
            vmax = max(vec) if len(vec) else 1.0
            return vec, f"Ativações de {self.current_layer_name} (x_ref={xidx})", (vmin, vmax)

        else:
            vec, src = pick_relevance_vector_for_layer(self.current_layer_name, self.layers_dict, prefer_self=True)
            if vec is None:
                return [], "Sem 'heatmap' compatível para esta camada.", (0.0, 1.0)
            vmin = min(vec) if len(vec) else -1.0
            vmax = max(vec) if len(vec) else 1.0
            title = f"Relevância (Grad×Act) p/ {self.current_layer_name}"
            if src and src != self.current_layer_name:
                title += f"  [derivada de: {src}]"
            return vec, title, (vmin, vmax)

    def compute_grid(self, n_items, canvas_w, canvas_h, diameter, padding):
        if n_items <= 0:
            return 0, 0, []
        cols = max(1, int(math.sqrt(n_items)))
        rows = math.ceil(n_items / cols)

        cell_w = diameter + padding
        cell_h = diameter + padding
        grid_w = cols * cell_w
        grid_h = rows * cell_h

        ox = max(0, (canvas_w - grid_w) // 2)
        oy = max(30, (canvas_h - grid_h) // 2)
        coords = []
        for i in range(n_items):
            r = i // cols
            c = i % cols
            x = ox + c * cell_w
            y = oy + r * cell_h
            coords.append((x, y))
        return rows, cols, coords

    def redraw(self):
        self.canvas.delete("all")
        values, title, (vmin, vmax) = self.get_values_for_current()

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        self.canvas.create_text(10, 10, anchor="w", fill="#dddddd", text=title, font=("Segoe UI", 12, "bold"))

        n = len(values)
        dia = max(6, int(self.circle_size_var.get()))
        pad = max(2, int(self.padding_var.get()))
        rows, cols, coords = self.compute_grid(n, cw, ch, dia, pad)

        thr = abs(float(self.threshold_var.get())) if isinstance(self.threshold_var.get(), (float, int)) else 0.0

        if self.mode_var.get() == "activation":
            cmap = val_to_hex_color_sequential
        else:
            cmap = val_to_hex_color_diverging

        for i, (x, y) in enumerate(coords):
            if i >= n: break
            v = values[i]
            color = cmap(v, vmin, vmax)
            x0, y0 = x, y
            x1, y1 = x + dia, y + dia
            self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="")
            if thr > 0 and abs(v) >= thr:
                self.canvas.create_oval(x0, y0, x1, y1, outline="#000000", width=2)
                self.canvas.create_oval(x0+1, y0+1, x1-1, y1-1, outline="#ffffff", width=1)

        if n > 0:
            self.draw_legend(vmin, vmax, 10, ch - 50, 200, 16, cmap)

        self.status.config(text=f"Neurônios: {n} | Min: {vmin:.4f}  Max: {vmax:.4f}")

    def draw_legend(self, vmin, vmax, x, y, w, h, cmap_func):
        steps = 64
        for i in range(steps):
            t = i / (steps - 1)
            v = vmin + t * (vmax - vmin)
            color = cmap_func(v, vmin, vmax)
            self.canvas.create_rectangle(x + int(t * w), y, x + int((i + 1) / (steps - 1) * w), y + h, outline="", fill=color)
        self.canvas.create_rectangle(x, y, x + w, y + h, outline="#222222")
        self.canvas.create_text(x, y - 4, anchor="sw", fill="#cccccc", text=f"{vmin:.2f}")
        self.canvas.create_text(x + w, y - 4, anchor="se", fill="#cccccc", text=f"{vmax:.2f}")
        self.canvas.create_text(x + w // 2, y + h + 2, anchor="n", fill="#cccccc", text="Intensidade")

if __name__ == "__main__":
    DenseViewerApp().mainloop()
