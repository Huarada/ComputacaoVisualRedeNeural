#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Feature Maps de Convolução (por camada/por amostra)
------------------------------------------------------------------
- Abre um JSON que contenha ativações de camadas convolucionais.
- Detecta automaticamente se os dados estão em (C,H,W) **ou** (N,C,H,W).
- Se for (N,C,H,W): permite escolher a **amostra**; sempre exibe um **mosaico único** com um
  intervalo de canais (Canal início, Qtd canais).
- Se for (C,H,W): exibe direto o mosaico dos canais.
- Normalização por mapa opcional; colormap de heatmap (inferno).
- **Atualização:** interface preparada para até **32 canais** por mosaico.
  (O limite superior do spin "Qtd canais" é min(32, C) para a camada selecionada.)

Compatível com esquemas comuns: chaves como 'conv.layers', 'map'/'maps', 'acts', 'feature_maps'.

Requisitos: Python 3.x, numpy, matplotlib (TkAgg)
Execute:  python visualizador_de_acts_maps_tkinter_matplotlib.py
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from tkinter import (
    Tk, Frame, Button, Label, filedialog, StringVar, IntVar,
    OptionMenu, Spinbox, Checkbutton, DISABLED, NORMAL, messagebox
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --------------------------- Utilitários --------------------------- #

def _np(x):
    try:
        return np.array(x, dtype=float)
    except Exception as e:
        raise ValueError(f"Falha ao converter dados em numpy array: {e}")


def _find_tensor_in_layer(layer_dict):
    """Tenta encontrar e padronizar um tensor de feature maps numa camada.
    Retorna (tensor, shape_tag), onde shape_tag é 'CHW' ou 'NCHW'.
    Heurística: procura por chaves usuais e também varre o dict.
    """
    candidates = []
    for key in [
        'acts', 'maps', 'map', 'feature_maps', 'features', 'data', 'activations'
    ]:
        if key in layer_dict:
            candidates.append(layer_dict[key])

    # fallback: varre valores que sejam listas/dicts numéricos
    if not candidates:
        for v in layer_dict.values():
            candidates.append(v)

    def is_numeric_nested(a):
        try:
            arr = _np(a)
            return arr.ndim in (3, 4)
        except Exception:
            return False

    for item in candidates:
        if isinstance(item, dict):
            # pode ser {'samples': [...]} ou {'0': [...]} etc.
            for v in item.values():
                if is_numeric_nested(v):
                    arr = _np(v)
                    if arr.ndim == 3:
                        return arr, 'CHW'
                    if arr.ndim == 4:
                        return arr, 'NCHW'
        else:
            if is_numeric_nested(item):
                arr = _np(item)
                if arr.ndim == 3:
                    return arr, 'CHW'
                if arr.ndim == 4:
                    return arr, 'NCHW'

    raise ValueError("Não encontrei tensor 3D/4D (CHW/NCHW) dentro da camada.")


def _coerce_CHW_or_NCHW(arr, H=None, W=None):
    """Ajusta shapes para CHW ou NCHW, aplicando dica de H/W se necessário."""
    a = _np(arr)
    if a.ndim == 4:
        # N,C,H,W ou N,H,W,C — tenta inferir pela ‘parede’ das dimensões
        N, d1, d2, d3 = a.shape
        # se últimos dois forem iguais a H/W quadrado, assume N,C,H,W já
        if d2 * d3 >= d1 * d2 and d2 >= 2 and d3 >= 2:
            return a, 'NCHW'
        # tenta permutar como NHWC->NCHW
        return np.transpose(a, (0, 3, 1, 2)), 'NCHW'
    if a.ndim == 3:
        d0, d1, d2 = a.shape
        # pode ser H,W,C -> permutar
        if d0 < 16 and (d1 >= 8 and d2 >= 8):
            # parece CHW
            return a, 'CHW'
        # se for H,W,C
        return np.transpose(a, (2, 0, 1)), 'CHW'
    if a.ndim == 2:
        # (C, H*W)
        if H and W and a.shape[1] == H*W:
            return a.reshape(a.shape[0], H, W), 'CHW'
        s = int(math.sqrt(a.shape[1]))
        if s*s == a.shape[1]:
            return a.reshape(a.shape[0], s, s), 'CHW'
    if a.ndim == 1:
        s = int(math.sqrt(a.size))
        if s*s == a.size:
            return a.reshape(1, s, s), 'CHW'
    raise ValueError("Não foi possível padronizar para CHW/NCHW.")


def normalize_maps(maps, per_map=True):
    m = maps.astype(float).copy()
    if per_map:
        # normaliza em cada canal individualmente
        flat = m.reshape(m.shape[0], -1)
        mins = flat.min(axis=1)[:, None, None]
        maxs = flat.max(axis=1)[:, None, None]
        m = (m - mins) / (maxs - mins + 1e-9)
    else:
        mn, mx = m.min(), m.max()
        m = (m - mn) / (mx - mn + 1e-9)
    return m


def make_mosaic(channels, H, W, start=0, count=None):
    """Monta um mosaico único a partir de canais (C,H,W)."""
    C = channels.shape[0]
    if count is None:
        count = C - start
    end = max(start, min(start + count, C))
    sel = channels[start:end]
    n = sel.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    mosaic = np.zeros((rows * H, cols * W), dtype=float)
    for i in range(n):
        r = i // cols
        c = i % cols
        mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = sel[i]
    return mosaic, start, end-1


# --------------------------- GUI --------------------------- #

class App:
    def __init__(self, master: Tk):
        self.master = master
        master.title("Feature Maps (Conv) — Mosaico Único")

        top = Frame(master)
        top.pack(fill="x", padx=8, pady=6)

        Button(top, text="Abrir JSON", command=self.on_open).pack(side="left")

        Label(top, text="Camada:").pack(side="left", padx=(12, 4))
        self.layer_var = StringVar(value="")
        self.layer_menu = OptionMenu(top, self.layer_var, "")
        self.layer_menu.pack(side="left")
        self.layer_var.trace_add('write', lambda *args: self.refresh())

        # Controles de AMOSTRA (para NCHW)
        Label(top, text="Amostra:").pack(side="left", padx=(12, 4))
        self.sample_var = IntVar(value=0)
        self.sample_spin = Spinbox(top, from_=0, to=9999, textvariable=self.sample_var, width=5, command=self.refresh)
        self.sample_spin.pack(side="left")

        # Controles de CANAL
        Label(top, text="Canal início:").pack(side="left", padx=(12, 4))
        self.ch_start_var = IntVar(value=0)
        self.ch_start_spin = Spinbox(top, from_=0, to=9999, textvariable=self.ch_start_var, width=5, command=self.refresh)
        self.ch_start_spin.pack(side="left")

        Label(top, text="Qtd canais:").pack(side="left", padx=(12, 4))
        self.ch_count_var = IntVar(value=32)  # padrão agora 32
        self.ch_count_spin = Spinbox(top, from_=1, to=32, textvariable=self.ch_count_var, width=5, command=self.refresh)
        self.ch_count_spin.pack(side="left")

        self.norm_var = IntVar(value=1)
        Checkbutton(top, text="Normalizar por canal", variable=self.norm_var, command=self.refresh).pack(side="left", padx=(12,0))

        # Figure/Canvas
        self.fig = Figure(figsize=(9.5, 6.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Estado
        self.raw = None
        self.layers = {}
        self.cache = {}
        self.layer_shapes = {}  # nome -> ('CHW'|'NCHW', (dims))

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
                self.raw = json.load(f)
        except Exception as e:
            messagebox.showerror("Erro ao abrir JSON", str(e))
            return

        # Extrai dicionário de camadas convolucionais
        layers = {}
        if isinstance(self.raw, dict):
            conv = self.raw.get('conv') if isinstance(self.raw.get('conv'), dict) else None
            if conv and isinstance(conv.get('layers'), dict):
                layers = conv['layers']
            else:
                # fallback: qualquer entrada com sinal de mapa
                layers = {k: v for k, v in self.raw.items() if isinstance(v, dict) and any(s in v for s in ('map','maps','acts','feature_maps','features'))}

        if not layers:
            messagebox.showerror("Erro", "Não encontrei camadas convolucionais no JSON.")
            return

        self.layers = layers
        self.cache.clear()
        self.layer_shapes.clear()

        # Prepara shapes por camada
        for name, info in self.layers.items():
            try:
                arr, tag = _find_tensor_in_layer(info)
                arr, tag = _coerce_CHW_or_NCHW(arr, H=info.get('H') or info.get('height'), W=info.get('W') or info.get('width'))
                self.cache[name] = arr  # guarda tensor padronizado
                self.layer_shapes[name] = (tag, arr.shape)
            except Exception:
                # ignora camadas sem tensor válido
                pass

        valid = list(self.cache.keys())
        if not valid:
            messagebox.showerror("Erro", "Nenhuma camada apresentou tensor CHW/NCHW reconhecido.")
            return

        # Popular menu
        menu = self.layer_menu['menu']
        menu.delete(0, 'end')
        first = None
        for name in valid:
            if first is None:
                first = name
            menu.add_command(label=name, command=lambda n=name: self.layer_var.set(n))
        self.layer_var.set(first)
        self.sample_var.set(0)
        self.ch_start_var.set(0)
        # ch_count será ajustado dinamicamente na refresh() de acordo com C (até 32)
        # mas já deixamos 32 por padrão

        self.refresh()

    def _update_channel_spin_limits(self, C: int):
        """Limita dinamicamente os spinboxes de acordo com a camada atual."""
        max_count = max(1, min(32, int(C)))  # novo limite superior = 32
        self.ch_count_spin.config(to=max_count)
        # garante que o valor atual respeite o limite
        if int(self.ch_count_var.get()) > max_count:
            self.ch_count_var.set(max_count)

        self.ch_start_spin.config(to=max(0, int(C) - 1))
        if int(self.ch_start_var.get()) > C - 1:
            self.ch_start_var.set(max(0, C - 1))

    def refresh(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.axis('off')

        name = self.layer_var.get()
        if not name or name not in self.cache:
            ax.text(0.5, 0.5, 'Abra um JSON e selecione a camada', ha='center', va='center')
            self.canvas.draw()
            return

        arr = self.cache[name]
        tag, shape = self.layer_shapes[name]

        # Ajusta UI de amostra conforme tag
        if tag == 'NCHW':
            self.sample_spin.config(state=NORMAL)
            N, C, H, W = shape
            s = max(0, min(int(self.sample_var.get()), N-1))
            self.sample_var.set(s)
            channels = arr[s]  # (C,H,W)
        else:
            self.sample_spin.config(state=DISABLED)
            C, H, W = shape
            channels = arr

        # Atualiza limites dos spinboxes de canais com base no C atual (máx 32)
        self._update_channel_spin_limits(C)

        # Normalização por canal
        channels = normalize_maps(channels, per_map=bool(self.norm_var.get()))

        ch_start = max(0, int(self.ch_start_var.get()))
        ch_count = max(1, int(self.ch_count_var.get()))
        mosaic, a, b = make_mosaic(channels, H, W, start=ch_start, count=ch_count)

        img = ax.imshow(mosaic, cmap='inferno', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"{name}  •  canais {a}–{b} (máx 32 por mosaico)"
        if tag == 'NCHW':
            title = f"{name}  •  amostra {self.sample_var.get()}  •  {title}"
        ax.set_title(title, fontsize=9)
        self.fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        self.canvas.draw()


if __name__ == '__main__':
    root = Tk()
    root.geometry('1000x720')
    app = App(root)
    root.mainloop()
