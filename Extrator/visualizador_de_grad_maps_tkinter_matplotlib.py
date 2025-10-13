#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Feature Maps (Heatmap em mosaico) para JSON
----------------------------------------------------------
- Abre um JSON com mapas de camadas convolucionais
- Escolhe a camada e um intervalo de mapas (Início, Qtd)
- Exibe **uma única imagem** (mosaico) com todos os mapas selecionados
- Normalização por mapa (opcional)

Requisitos: Python 3.x, numpy, matplotlib (TkAgg)
Execute:  python visualizador_maps.py
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from tkinter import (
    Tk, Frame, Button, Label, filedialog, StringVar, IntVar,
    OptionMenu, Spinbox, Checkbutton, messagebox
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --------------------------- Utilitários --------------------------- #

def _to_numpy(x):
    try:
        return np.array(x, dtype=float)
    except Exception as e:
        raise ValueError(f"Falha ao converter dados do JSON em numpy array: {e}")


def infer_maps(arr, H=None, W=None, count=None):
    """Tenta organizar os dados em shape (N, H, W)."""
    a = _to_numpy(arr)

    # Se já estiver (N,H,W)
    if a.ndim == 3:
        maps = a
        H_ = a.shape[1]
        W_ = a.shape[2]

    # Se (H,W)
    elif a.ndim == 2:
        H_ = a.shape[0] if H is None else H
        W_ = a.shape[1] if W is None else W
        if H_ * W_ == a.size and a.shape == (H_, W_):
            maps = a.reshape(1, H_, W_)
        else:
            # Tenta (N, H*W)
            s = int(math.sqrt(a.shape[1]))
            if s * s == a.shape[1]:
                H_, W_ = s, s
                maps = a.reshape(a.shape[0], H_, W_)
            else:
                raise ValueError("Não consegui inferir (N,H,W) a partir de uma matriz 2D.")

    # Se vetor 1D -> supõe imagem quadrada única
    elif a.ndim == 1:
        s = int(math.sqrt(a.size))
        if s * s != a.size:
            raise ValueError("Vetor 1D não é quadrado perfeito; informe H e W.")
        H_, W_ = s, s
        maps = a.reshape(1, H_, W_)

    else:
        raise ValueError("Dimensionalidade não suportada para os mapas.")

    # Aplica dica externa de H/W, se vier
    if H is not None and W is not None and (H_ != H or W_ != W):
        if maps.reshape(maps.shape[0], -1).shape[1] != H * W:
            raise ValueError("H/W fornecidos não batem com o tamanho dos mapas.")
        maps = maps.reshape(maps.shape[0], H, W)
        H_, W_ = H, W

    # Respeita 'count' (se vier maior, corta; se menor, mantém)
    if count is not None and maps.shape[0] > int(count):
        maps = maps[:int(count)]

    return maps.astype(float), int(H_), int(W_)


def normalize_maps(maps, per_map=True):
    """Normaliza para [0,1]. Se per_map=True, normaliza cada mapa individualmente."""
    m = maps.astype(float).copy()
    if m.size == 0:
        return m
    if per_map:
        for i in range(m.shape[0]):
            mn = float(m[i].min())
            mx = float(m[i].max())
            m[i] = (m[i] - mn) / (mx - mn + 1e-9)
    else:
        mn = float(m.min())
        mx = float(m.max())
        m = (m - mn) / (mx - mn + 1e-9)
    return m


# --------------------------- GUI --------------------------- #

class App:
    def __init__(self, master: Tk):
        self.master = master
        master.title("Visualizador de Heatmap (Mosaico Único)")

        # Top bar
        top = Frame(master)
        top.pack(fill="x", padx=8, pady=6)

        Button(top, text="Abrir JSON", command=self.on_open).pack(side="left")

        Label(top, text="Camada:").pack(side="left", padx=(12, 4))
        self.layer_var = StringVar(value="")
        self.layer_menu = OptionMenu(top, self.layer_var, "")
        self.layer_menu.pack(side="left")
        self.layer_var.trace_add('write', lambda *args: self.refresh())

        Label(top, text="Início:").pack(side="left", padx=(12, 4))
        self.start_var = IntVar(value=0)
        Spinbox(top, from_=0, to=9999, textvariable=self.start_var, width=6, command=self.refresh).pack(side="left")

        Label(top, text="Qtd:").pack(side="left", padx=(12, 4))
        self.qtd_var = IntVar(value=16)
        Spinbox(top, from_=1, to=512, textvariable=self.qtd_var, width=4, command=self.refresh).pack(side="left")

        self.norm_var = IntVar(value=1)
        Checkbutton(top, text="Normalizar por mapa", variable=self.norm_var, command=self.refresh).pack(side="left", padx=(12,0))

        # Figure/Canvas
        self.fig = Figure(figsize=(9, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Estado
        self.data = None
        self.layers = {}
        self.cache = {}

    # --------------- Ações --------------- #
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Selecione o arquivo JSON",
            filetypes=[("JSON files", "*.json"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            messagebox.showerror("Erro ao abrir JSON", str(e))
            return

        # Tenta no formato conv.layers, senão varre raiz
        layers = {}
        if isinstance(self.data, dict):
            conv = self.data.get('conv') if isinstance(self.data.get('conv'), dict) else None
            if conv and isinstance(conv.get('layers'), dict):
                layers = conv['layers']
            else:
                # fallback: qualquer dict com chave 'map'
                layers = {k: v for k, v in self.data.items() if isinstance(v, dict) and 'map' in v}

        if not layers:
            messagebox.showerror("Erro", "Nenhuma camada com chave 'map' encontrada no JSON.")
            return

        self.layers = layers
        self.cache.clear()

        # Popular menu de camadas
        menu = self.layer_menu['menu']
        menu.delete(0, 'end')
        first = None
        for name in self.layers.keys():
            if first is None:
                first = name
            menu.add_command(label=name, command=lambda n=name: self.layer_var.set(n))
        self.layer_var.set(first or "")

        self.start_var.set(0)
        self.qtd_var.set(16)
        self.refresh()

    def get_maps_for_layer(self, name):
        if name in self.cache:
            return self.cache[name]
        info = self.layers.get(name, {})
        H = info.get('H') or info.get('height')
        W = info.get('W') or info.get('width')
        count = info.get('count') or info.get('n') or info.get('num')
        arr = info.get('map') or info.get('maps') or info.get('feature_map')
        if arr is None:
            raise ValueError(f"Camada '{name}' não possui chave 'map'.")
        maps, H, W = infer_maps(arr, H=H, W=W, count=count)
        self.cache[name] = (maps, H, W)
        return self.cache[name]

    def refresh(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.axis('off')

        if not self.layers or not self.layer_var.get():
            ax.text(0.5, 0.5, 'Abra um JSON para visualizar', ha='center', va='center')
            self.canvas.draw()
            return

        try:
            maps, H, W = self.get_maps_for_layer(self.layer_var.get())
        except Exception as e:
            ax.text(0.5, 0.5, f'Erro: {e}', ha='center', va='center', wrap=True)
            self.canvas.draw()
            return

        start = max(0, int(self.start_var.get()))
        qtd = max(1, int(self.qtd_var.get()))
        end = min(start + qtd, maps.shape[0])
        if start >= end:
            ax.text(0.5, 0.5, 'Índice fora do intervalo.', ha='center', va='center')
            self.canvas.draw()
            return

        sel = maps[start:end]
        sel = normalize_maps(sel, per_map=bool(self.norm_var.get()))
        n = sel.shape[0]

        # Define grid para o mosaico único
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        mosaic = np.zeros((rows * H, cols * W), dtype=float)
        for i in range(n):
            r = i // cols
            c = i % cols
            mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = sel[i]

        img = ax.imshow(mosaic, cmap='inferno', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{self.layer_var.get()} • mapas {start}–{end-1}", fontsize=9)
        self.fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        self.canvas.draw()


if __name__ == '__main__':
    root = Tk()
    root.geometry('980x720')
    app = App(root)
    root.mainloop()
