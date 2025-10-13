from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Any
import os, json, random, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ==========================================================
# Utilidades
# ==========================================================

def _minmax_norm(x: Tensor, eps: float = 1e-12) -> Tensor:
    x_min = x.min()
    x_max = x.max()
    rng = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / rng

# ==========================================================
# PROBES para modo "amostral" (seleção por posições globais)
# ==========================================================

@dataclass
class _SamplePack:
    global_pos: int
    x_in: Tensor   # (in_features,)
    y_out: Tensor  # (out_features,)

class _DenseProbeSampled:
    """
    Captura por amostras selecionadas por posição global:
      - x_in  (entrada do nn.Linear)
      - y_out (saída do nn.Linear)
      - dL/dy (grad da perda w.r.t. saída do nn.Linear)

    Heatmap: média sobre amostras de ReLU((dY @ W) * X) com min-max.
    """
    def __init__(self, layer: nn.Linear, name: str):
        self.layer = layer
        self.name = name
        self.hook_fwd = None
        self._xs: List[Tensor] = []           # (N, in_features)
        self._dys: List[Tensor] = []          # (N, out_features)
        self._samples: List[_SamplePack] = [] # ativações por amostra (para JSON)

    def clear(self):
        self._xs.clear()
        self._dys.clear()
        self._samples.clear()

    def register(self, batch_indices_supplier, batch_global_window_supplier):
        if self.hook_fwd is not None:
            self.hook_fwd.remove()
            self.hook_fwd = None

        def fwd_hook(_mod, inputs, output):
            x_in: Tensor = inputs[0]
            if x_in.dim() != 2:
                x_in = x_in.view(x_in.size(0), -1)  # [B, in_features]

            to_take: List[int] = list(batch_indices_supplier()) or []
            if not to_take:
                return
            to_take = [i for i in to_take if 0 <= i < x_in.size(0)]
            if not to_take:
                return

            xs_sel = x_in.detach()[to_take].cpu()   # (k, in_features)
            ys_sel = output.detach()[to_take].cpu() # (k, out_features)

            start, _end = batch_global_window_supplier()
            for j, i_local in enumerate(to_take):
                gpos = start + i_local
                self._samples.append(_SamplePack(global_pos=int(gpos),
                                                 x_in=xs_sel[j].clone(),
                                                 y_out=ys_sel[j].clone()))

            def _bwd_hook(grad_out: Tensor):
                self._xs.append(xs_sel.clone())
                self._dys.append(grad_out.detach()[to_take].cpu())
                return grad_out
            output.register_hook(_bwd_hook)

        self.hook_fwd = self.layer.register_forward_hook(fwd_hook)

    def remove(self):
        if self.hook_fwd is not None:
            self.hook_fwd.remove()
            self.hook_fwd = None

    @torch.no_grad()
    def aggregate_relevance(self, device: Optional[torch.device] = None) -> Optional[Tensor]:
        if len(self._xs) == 0:
            return None
        X  = torch.cat(self._xs,  dim=0)      # (N, in)
        dY = torch.cat(self._dys, dim=0)      # (N, out)
        W  = self.layer.weight.detach().cpu() # (out, in)
        R  = (dY @ W) * X                     # (N, in)
        r  = F.relu(R.mean(dim=0))            # (in,)
        if torch.all(r == 0):
            return r.to(device) if device is not None else r
        r  = _minmax_norm(r)
        return r.to(device) if device is not None else r

    @torch.no_grad()
    def json_block(self, include_per_sample_relevance: bool = False) -> dict:
        in_f  = int(self.layer.in_features)
        out_f = int(self.layer.out_features)

        samples = []
        for s in sorted(self._samples, key=lambda t: t.global_pos):
            entry = {
                "global_pos": int(s.global_pos),
                "x_in": s.x_in.tolist(),
                "y_out": s.y_out.tolist(),
            }
            samples.append(entry)

        block = {
            "in_features": in_f,
            "out_features": out_f,
            "ref_inputs": [],
            "ref_acts": [],
            "samples": samples,
        }

        if include_per_sample_relevance and len(self._xs) > 0:
            X  = torch.cat(self._xs,  dim=0)
            dY = torch.cat(self._dys, dim=0)
            W  = self.layer.weight.detach().cpu()
            R  = (dY @ W) * X
            R = F.relu(R)
            Rs = []
            for i in range(R.size(0)):
                r_i = R[i]
                if torch.all(r_i == 0):
                    Rs.append([0.0] * r_i.numel())
                else:
                    r_i = _minmax_norm(r_i)
                    Rs.append(r_i.tolist())
            block["per_sample_relevance"] = Rs

        return block

# ==========================================================
# PROBES para modo "ref" (entradas fixas)
# ==========================================================

class _DenseProbeRef:
    """
    Modo REF: salva ativações para um conjunto fixo de entradas e acumula
    heatmap médio usando **Gradient × Activation** (g * x) durante os backward da época.
    """
    def __init__(self, layer: nn.Linear, name: str, max_features: Optional[int] = None):
        self.layer = layer
        self.name = name
        self.max_features = max_features

        self._ref_in: List[Tensor] = []   # lista de (B, in)
        self._ref_out: List[Tensor] = []  # lista de (B, out)

        self._heat_sum: Optional[Tensor] = None  # (in,)
        self._heat_count: int = 0
        self._bwd_handle: Optional[Any] = None
        self._fwd_in_handle: Optional[Any] = None
        self._last_x: Optional[Tensor] = None  # (B, in) do último forward

    def reset_epoch(self):
        fin = int(self.layer.in_features)
        if (self.max_features is not None) and (fin > self.max_features):
            fin = self.max_features
        self._ref_in.clear(); self._ref_out.clear()
        self._heat_sum = torch.zeros(fin, dtype=torch.float32, device="cpu")
        self._heat_count = 0
        self._last_x = None
    def capture_refs(self, x: Tensor):
        # hooks temporários para capturar entradas/saídas em uma passada com no_grad
        handles = []
        def _in(_m, inputs):
            xin = inputs[0].detach()
            if xin.dim() > 2:
                xin = xin.view(xin.size(0), -1)
            if self.max_features is not None and xin.size(-1) > self.max_features:
                xin = xin[..., : self.max_features]
            self._ref_in.append(xin.cpu())
        def _out(_m, inputs, outputs):
            y = outputs.detach()
            if y.dim() > 2:
                y = y.view(y.size(0), -1)
            if self.max_features is not None and y.size(-1) > self.max_features:
                y = y[..., : self.max_features]
            self._ref_out.append(y.cpu())
        handles.append(self.layer.register_forward_pre_hook(_in))
        handles.append(self.layer.register_forward_hook(_out))
        return handles

    def register_bwd(self):
        # hook de entrada para capturar x do mesmo forward cujos grad_input serão recebidos no backward
        def _in_hook(_m, inputs):
            xin = inputs[0].detach()
            if xin.dim() > 2:
                xin = xin.view(xin.size(0), -1)
            if self.max_features is not None and xin.size(-1) > self.max_features:
                xin = xin[..., : self.max_features]
            self._last_x = xin
        self._fwd_in_handle = self.layer.register_forward_pre_hook(_in_hook)

        def bwd_hook(module, grad_input, grad_output):
            if not grad_input or grad_input[0] is None or self._last_x is None:
                return
            g = grad_input[0]  # (B, Fin)
            if g.dim() > 2:
                g = g.view(g.size(0), -1)
            if self.max_features is not None and g.size(-1) > self.max_features:
                g = g[..., : self.max_features]
            # Gradient × Activation (médio no batch), ReLU para relevância positiva
            x = self._last_x.to(g.device)
            gx = (g.detach() * x.detach()).mean(dim=0).clamp_min(0.0).cpu()
            self._heat_sum += gx
            self._heat_count += 1
        self._bwd_handle = self.layer.register_full_backward_hook(bwd_hook)

    def remove_bwd(self):
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        if self._fwd_in_handle is not None:
            self._fwd_in_handle.remove()
            self._fwd_in_handle = None
    @torch.no_grad()
    def json_block_and_heat(self) -> Tuple[dict, List[float]]:
        xin = torch.cat(self._ref_in, dim=0) if len(self._ref_in) else torch.empty(0)
        yout = torch.cat(self._ref_out, dim=0) if len(self._ref_out) else torch.empty(0)

        h = self._heat_sum.clone()
        if self._heat_count > 0:
            h = h / float(self._heat_count)
            if h.numel():
                h_min, h_max = float(h.min().item()), float(h.max().item())
                h = (h - h_min) / (h_max - h_min) if h_max > h_min else torch.zeros_like(h)
                heatmap = h.tolist() if h.numel() else []

        block = {
            "in_features": int(self.layer.in_features),
            "out_features": int(self.layer.out_features),
            "ref_inputs": xin.tolist() if xin.numel() else [],
            "ref_acts":   yout.tolist() if yout.numel() else [],
        }
        return block, heatmap

# ==========================================================
# Agregadores unificados
# ==========================================================

class DenseCAMAggregatorSampled:
    """
    Agregador estilo CAM para nn.Linear com *amostras* (posições globais).
    JSON por época é **unificado** com o modo REF: inclui sempre campos
    "ref_inputs" e "ref_acts" (vazios neste modo) e "samples" com global_pos.
    """
    def __init__(
        self,
        model: nn.Module,
        layer_filter: Optional[Iterable[str] | str | callable] = None,
        max_samples_per_epoch: int = 10,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        include_per_sample_relevance: bool = False,
    ):
        self.model = model
        self.max_samples = max_samples_per_epoch
        self.device = device
        self.include_per_sample_relevance = include_per_sample_relevance
        self.rng = random.Random(seed) if seed is not None else random

        self.probes: Dict[str, _DenseProbeSampled] = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if self._match_layer(name, m, layer_filter):
                    self.probes[name] = _DenseProbeSampled(m, name)
        if not self.probes:
            raise ValueError("Nenhum nn.Linear encontrado (ou filtrado).")

        self._epoch_positions: List[int] = []
        self._batch_window: Tuple[int, int] = (0, 0)

        def supplier_locals():
            start, end = self._batch_window
            if start == end:
                return []
            gset = set(self._epoch_positions)
            return [i for i, g in enumerate(range(start, end)) if g in gset]

        def supplier_window():
            return self._batch_window

        for p in self.probes.values():
            p.register(batch_indices_supplier=supplier_locals,
                       batch_global_window_supplier=supplier_window)

    @staticmethod
    def _match_layer(name: str, module: nn.Module, layer_filter) -> bool:
        if layer_filter is None:
            return True
        if isinstance(layer_filter, str):
            return layer_filter in name
        if isinstance(layer_filter, Iterable) and not callable(layer_filter):
            return name in set(layer_filter)
        if callable(layer_filter):
            return bool(layer_filter(name, module))
        return False

    def begin_epoch(self, total_examples: int):
        k = min(self.max_samples, total_examples)
        self._epoch_positions = sorted(self.rng.sample(range(total_examples), k))
        for p in self.probes.values():
            p.clear()

    def on_batch_begin(self, batch_index: int, batch_size: int):
        start = batch_index * batch_size
        end = start + batch_size
        self._batch_window = (start, end)

    @torch.no_grad()
    def end_epoch(self, save_dir: str, epoch: int, run_id: str = "default") -> str:
        os.makedirs(save_dir, exist_ok=True)
        payload = {
            "meta": {
                "epoch": int(epoch),
                "run_id": str(run_id),
                "timestamp": int(time.time()),
                "selected_positions": [int(p) for p in self._epoch_positions],
            },
            "layers": {}
        }

        for name, p in self.probes.items():
            block = p.json_block(include_per_sample_relevance=self.include_per_sample_relevance)
            r = p.aggregate_relevance(device=self.device)
            block["heatmap"] = [] if r is None else r.detach().cpu().tolist()
            payload["layers"][name] = block

        fname = os.path.join(save_dir, f"dense_epoch_{epoch:04d}_{run_id}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fname

    def selected_positions(self) -> List[int]:
        return list(self._epoch_positions)

    def remove_hooks(self):
        for p in self.probes.values():
            p.remove()

class DenseCAMAggregatorRef:
    """
    Agregador para nn.Linear com conjunto *fixo* de entradas de referência.
    - Salva ref_inputs/ref_acts por camada usando uma passada em no_grad.
    - Acumula heatmap médio via |grad_input| durante os backward da época.
    JSON é o mesmo do modo amostral; "selected_positions" fica vazio.
    """
    def __init__(
        self,
        model: nn.Module,
        layer_filter: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        max_features: Optional[int] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.layer_filter = layer_filter
        self.max_features = max_features

        self._linear_layers: Dict[str, nn.Linear] = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                if (self.layer_filter is None) or (self.layer_filter in name):
                    self._linear_layers[name] = mod

        self._epoch_total = 0
        self._ref_inputs_batch: Optional[torch.Tensor] = None
        self._probes: Dict[str, _DenseProbeRef] = {n: _DenseProbeRef(m, n, max_features) for n, m in self._linear_layers.items()}

    def set_ref_inputs(self, ref_inputs: torch.Tensor, max_samples: Optional[int] = 10, max_features: Optional[int] = None) -> None:
        if max_samples is not None:
            ref_inputs = ref_inputs[:max_samples]
        self._ref_inputs_batch = ref_inputs.detach().to(self.device)
        if max_features is not None:
            self.max_features = max_features
            for p in self._probes.values():
                p.max_features = max_features

    def begin_epoch(self, total_examples: int) -> None:
        self._epoch_total = int(total_examples)
        for p in self._probes.values():
            p.reset_epoch()
            p.remove_bwd()
            p.register_bwd()

        if self._ref_inputs_batch is not None:
            # Captura ref_inputs/ref_acts via passada no_grad
            was_training = self.model.training
            handles_all = []
            for p in self._probes.values():
                handles_all += p.capture_refs(self._ref_inputs_batch)
            self.model.eval()
            with torch.no_grad():
                _ = self.model(self._ref_inputs_batch)
            if was_training:
                self.model.train()
            for h in handles_all:
                h.remove()

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def end_epoch(self, save_dir: str, epoch: int, run_id: str = "default") -> str:
        for p in self._probes.values():
            p.remove_bwd()

        os.makedirs(save_dir, exist_ok=True)
        payload: Dict[str, Any] = {
            "meta": {"epoch": int(epoch), "timestamp": int(time.time()), "run_id": str(run_id),
                      "total_examples": int(self._epoch_total), "selected_positions": []},
            "layers": {}
        }

        for name, p in self._probes.items():
            block, heatmap = p.json_block_and_heat()
            block["heatmap"] = heatmap
            payload["layers"][name] = block

        out_path = os.path.join(save_dir, f"dense_epoch_{epoch:04d}_{run_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

# ==========================================================
# Wrapper conveniente para alternar entre modos
# ==========================================================

class DenseCAMAggregator:
    """
    Interface única:
      - mode="sampled": usa DenseCAMAggregatorSampled (seleciona posições globais)
      - mode="ref":     usa DenseCAMAggregatorRef (entradas fixas via set_ref_inputs)

    Todos os modos salvam JSON com o mesmo formato-base:
    {
      "meta": { epoch, run_id, timestamp, selected_positions },
      "layers": {
         <name>: {
           in_features, out_features,
           ref_inputs: [...],  # vazio em mode=sampled
           ref_acts:   [...],  # vazio em mode=sampled
           samples:    [...],  # ref: {kind, idx, x_in, y_out} | sampled: {global_pos, ...}
           heatmap:    [...],  # vetor in_features normalizado 0-1
           // opcional: per_sample_relevance
         }, ...
      }
    }
    """
    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def sampled(cls,
                model: nn.Module,
                layer_filter: Optional[Iterable[str] | str | callable] = None,
                max_samples_per_epoch: int = 10,
                device: Optional[torch.device] = None,
                seed: Optional[int] = None,
                include_per_sample_relevance: bool = False):
        return cls(DenseCAMAggregatorSampled(model, layer_filter, max_samples_per_epoch, device, seed, include_per_sample_relevance))

    @classmethod
    def ref(cls,
            model: nn.Module,
            layer_filter: Optional[str] = None,
            device: Optional[torch.device | str] = None,
            max_features: Optional[int] = None):
        return cls(DenseCAMAggregatorRef(model, layer_filter, device, max_features))

    # delegações
    def begin_epoch(self, total_examples: int):
        return self._impl.begin_epoch(total_examples)

    def on_batch_begin(self, *args, **kwargs):
        return self._impl.on_batch_begin(*args, **kwargs)

    def end_epoch(self, save_dir: str, epoch: int, run_id: str = "default") -> str:
        return self._impl.end_epoch(save_dir, epoch, run_id)

    # Métodos específicos de cada modo que podem (ou não) existir
    def set_ref_inputs(self, *args, **kwargs):
        if hasattr(self._impl, "set_ref_inputs"):
            return getattr(self._impl, "set_ref_inputs")(*args, **kwargs)
        raise AttributeError("set_ref_inputs só está disponível em mode='ref'.")

    def selected_positions(self) -> List[int]:
        if hasattr(self._impl, "selected_positions"):
            return getattr(self._impl, "selected_positions")()
        return []

    def remove_hooks(self):
        if hasattr(self._impl, "remove_hooks"):
            return getattr(self._impl, "remove_hooks")()
        return None