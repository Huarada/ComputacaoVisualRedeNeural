"""
Módulo unificado para **Layer-CAM** em camadas convolucionais com captura de
ativações de um conjunto fixo de entradas de referência no início de cada época.

Este arquivo substitui o Grad-CAM clássico pelo **Layer-CAM**:
  - Em vez de pesos globais por canal (média espacial do gradiente), usamos um
    *masking* ponto-a-ponto com o gradiente positivo: `ReLU(grad) ⊙ ativação`.
  - O heatmap de camada é a soma sobre os canais: `Σ_k ReLU(grad_k) ⊙ A_k`.
  - Mantemos ReLU final + normalização min–max por amostra.

API e formato de saída permanecem **compatíveis** com a versão anterior:
  - Classe principal: `ConvCAMAggregator`
  - Alias (compat): `ConvCAMAggregatorRef`
  - JSON por época via `export_epoch_json(...)` mantendo os mesmos campos.

Saída JSON por `export_epoch_json(epoch, path=None)`:
{
  "epoch": int,
  "layers": {
    name: {
      "H": int, "W": int, "count": int, "map": [[...]],
      "acts_meta": {"H":int,"W":int,"channels":int,"imgs":int},
      "acts": [ [ [ [ ... ] ] ] ]  # formato (B, C, H, W)
    }, ...
  }
}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- utilidades gerais ---------------------------- #

def _to_tuple_hw(x: Optional[Union[int, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    if x is None:
        return None
    if isinstance(x, int):
        return (x, x)
    return x


def _safe_interpolate(t: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Redimensiona t se um size (H, W) for dado; caso contrário retorna t."""
    if size is None:
        return t
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)


def _minmax_norm(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalização min–max por amostra no plano espacial (H×W) para cada mapa."""
    m_min = m.amin(dim=(-2, -1), keepdim=True)
    m_max = m.amax(dim=(-2, -1), keepdim=True)
    return (m - m_min) / (m_max - m_min + eps)


# --------------------------- Probe para uma camada ------------------------- #

class _LayerProbe:
    """Registra ativação e gradiente de *saída* de um módulo Conv2d."""

    def __init__(self, module: nn.Module, name: str):
        self.module = module
        self.name = name
        self.act: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self._fwd = None
        self._has_backward_hook = False
        self.enabled = True
        self.register()

    def register(self) -> None:
        def fwd_hook(_mod, _inp, out):
            if not self.enabled:
                return
            # out: (B, C, H, W)
            self.act = out.detach()

            # registra hook no próprio tensor de saída para pegar grad no backward
            def bwd_hook(gout):
                # gout: d(Loss)/d(out)
                self.grad = gout.detach()
            if hasattr(out, "register_hook") and out.requires_grad:
                out.register_hook(bwd_hook)
                self._has_backward_hook = True

        self._fwd = self.module.register_forward_hook(fwd_hook)

    def clear_batch(self) -> None:
        self.act = None
        self.grad = None

    def remove(self) -> None:
        if self._fwd is not None:
            self._fwd.remove()
            self._fwd = None
        self._has_backward_hook = False


# ---------------------- Agregador multi-camada por época ------------------- #

@dataclass
class _LayerAccumulators:
    sum_pos: Optional[torch.Tensor] = None  # (1, H, W)
    count: int = 0


class ConvCAMAggregator:
    """Orquestra hooks e agregação de heatmaps por época (**Layer-CAM**).

    Decisões simplificadas:
    - Sempre usa gradiente da Loss; sempre normaliza por amostra; ReLU (positivos).
    - Agrega por média incremental.

    Captura de ativações (entradas de referência):
    - Defina com `set_ref_inputs(...)`. No `begin_epoch()`, executa um forward
      sem grad apenas nessas entradas e guarda as ativações.
    """

    def __init__(
        self,
        model: nn.Module,
        select: Union[str, Iterable[str], Any] = "conv2d_all",
        out_size: Optional[Union[int, Tuple[int, int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        try:
            self.device = torch.device(device) if device is not None else next(model.parameters()).device
        except StopIteration:
            # modelo pode não ter parâmetros (edge case)
            self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.out_size = _to_tuple_hw(out_size)

        # --- Referências fixas para capturar ATIVAÇÕES por época ---
        self.ref_inputs: Optional[torch.Tensor] = None      # (B*, C, H, W) no device do modelo
        self.ref_max_images: int = 10                       # até 10 imagens
        self.ref_max_channels: int = 8                      # até 8 canais salvos por camada
        self.ref_act_size: Tuple[int, int] = (32, 32)       # resolução das ativações salvas
        self._ref_acts_raw: Dict[str, torch.Tensor] = {}    # (B, C, H, W) normalizado

        self.probes: Dict[str, _LayerProbe] = {}
        self.acc: Dict[str, _LayerAccumulators] = {}
        self._select_and_register(select)

    # ----------------------------- seleção de camadas ----------------------------- #

    def _select_and_register(self, select: Union[str, Iterable[str], Any]) -> None:
        candidates: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if name == "":
                continue  # ignora root
            candidates.append((name, module))

        def pick(name: str, m: nn.Module) -> bool:
            if isinstance(select, str) and select == "conv2d_all":
                return isinstance(m, nn.Conv2d)
            if callable(select):
                return bool(select(name, m))
            if isinstance(select, Iterable):
                return name in select
            return False

        for name, m in candidates:
            if pick(name, m):
                self.probes[name] = _LayerProbe(m, name)
                self.acc[name] = _LayerAccumulators()

        if not self.probes:
            raise ValueError("Nenhuma camada selecionada para monitoramento.")

    # ------------------------------- ciclo de época ------------------------------ #

    def begin_epoch(self) -> None:
        # Reinicia contadores de época; probes permanecem registrados
        for name, acc in self.acc.items():
            acc.sum_pos = None
            acc.count = 0
        for p in self.probes.values():
            p.clear_batch()
        # limpa e captura ativações das entradas de referência (se houver)
        self._ref_acts_raw = {}
        if self.ref_inputs is not None:
            self._capture_ref_activations()

    # ---------------------- entradas de referência (ativações) -------------------- #

    def set_ref_inputs(
        self,
        x: torch.Tensor,
        max_images: int = 10,
        max_channels: int = 8,
        act_size: Optional[Union[int, Tuple[int, int]]] = 32,
    ) -> None:
        """Define o conjunto fixo (até 10) de entradas para salvar **ativações** por época.
        - x: tensor (B, C, H, W). Será movido para o device do modelo se necessário.
        - max_images: máximo de imagens usadas (default 10).
        - max_channels: máximo de canais por camada (default 8, primeiros canais).
        - act_size: redimensiona ativações para este tamanho (int -> quadrado ou (H,W)).
        """
        if x.device != self.device:
            x = x.to(self.device)
        self.ref_inputs = x[:max_images].detach()
        self.ref_max_images = int(max_images)
        self.ref_max_channels = int(max_channels)
        self.ref_act_size = _to_tuple_hw(act_size) if act_size is not None else None

    @torch.no_grad()
    def _capture_ref_activations(self) -> None:
        """Faz um forward **somente nas entradas de referência** e guarda ativações.
        Mantém no máximo `ref_max_channels` canais por camada e redimensiona para `ref_act_size`.
        """
        if self.ref_inputs is None:
            return
        was_training = self.model.training
        self.model.eval()
        _ = self.model(self.ref_inputs)
        # após o forward, os probes têm `act`
        for name, probe in self.probes.items():
            act = probe.act
            if act is None:
                continue
            # recorta imagens e canais
            B = min(self.ref_max_images, act.size(0))
            C = min(self.ref_max_channels, act.size(1))
            sample = act[:B, :C].detach()
            # redimensiona espacialmente (opera em (B,C,H,W) diretamente)
            sample = _safe_interpolate(sample, self.ref_act_size)
            # normaliza por amostra (min–max em H×W) para cada (B,C)
            sample = _minmax_norm(sample)
            self._ref_acts_raw[name] = sample.cpu().float()  # (B, C, H, W)
        if was_training:
            self.model.train()
        # não limpamos os probes aqui; o loop de treino fará clear_batch() por batch

    # ---------------------- passo de agregação por batch ------------------------- #

    @torch.no_grad()
    def update_from_hooks(self) -> None:
        """Calcula **Layer-CAM** de cada probe e acumula média por época.

        Fórmula (por amostra): CAM = Σ_k ReLU(grad_k) ⊙ A_k
        Em seguida aplicamos ReLU e normalização min–max por amostra.
        """
        for name, probe in self.probes.items():
            act = probe.act
            grad = probe.grad
            if act is None or grad is None:
                continue

            if act.device != self.device:
                act = act.to(self.device)
            if grad.device != self.device:
                grad = grad.to(self.device)

            # Máscara positiva de gradiente (Layer-CAM)
            gpos = F.relu(grad)  # (B, C, H, W)

            # CAM por amostra: soma dos canais do produto ponto-a-ponto
            cam = (gpos * act).sum(dim=1)  # (B, H, W)

            # Mantém apenas contribuições positivas e normaliza por amostra
            cam_pos = F.relu(cam)
            cam_pos = _safe_interpolate(cam_pos.unsqueeze(1), self.out_size).squeeze(1)
            cam_pos = _minmax_norm(cam_pos)

            # Agregação por **média incremental**
            acc = self.acc[name]
            B, H, W = cam_pos.shape
            batch_mean = cam_pos.mean(dim=0, keepdim=True)  # (1, H, W)
            if acc.sum_pos is None or acc.count == 0:
                acc.sum_pos = batch_mean
            else:
                acc.sum_pos = acc.sum_pos + (batch_mean - acc.sum_pos) * (B / (acc.count + B))
            acc.count += B

            probe.clear_batch()

    # -------------------------- exportação/serialização ------------------------- #

    @torch.no_grad()
    def export_epoch_json(self, epoch: int, path: Optional[str] = None) -> Dict[str, Any]:
        """Exporta um JSON por época (um mapa positivo por camada + ativações de referência).
        Formato:
        {
          "epoch": int,
          "layers": {
            name: {
              "H": int, "W": int, "count": int, "map": [[...]],
              "acts_meta": {"H":int, "W":int, "channels":int, "imgs":int},
              "acts": [ [ [ [ ... ] ] ] ]  # lista (B, C, H, W)
            }
          }
        }
        """
        payload: Dict[str, Any] = {"epoch": int(epoch), "layers": {}}
        for name, acc in self.acc.items():
            if acc.sum_pos is None:
                continue
            H, W = int(acc.sum_pos.shape[-2]), int(acc.sum_pos.shape[-1])
            m = acc.sum_pos.clone()
            if m.numel() > 0:
                m = _minmax_norm(m)
            m = m.squeeze(0).cpu().float().tolist()
            layer_obj: Dict[str, Any] = {"H": H, "W": W, "count": int(acc.count), "map": m}
            # anexa ativações de referência se existirem
            if name in self._ref_acts_raw:
                sample = self._ref_acts_raw[name]  # (B, C, H, W)
                B, C, HH, WW = sample.shape
                layer_obj["acts_meta"] = {"H": int(HH), "W": int(WW), "channels": int(C), "imgs": int(B)}
                # lista com B entradas (cada uma [C][H][W])
                layer_obj["acts"] = [sample[i].tolist() for i in range(B)]
            else:
                layer_obj["acts_meta"] = {"H": 0, "W": 0, "channels": 0, "imgs": 0}
                layer_obj["acts"] = []
            payload["layers"][name] = layer_obj
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    @torch.no_grad()
    def export_epoch_json_lite(self, epoch: int, path: Optional[str] = None) -> Dict[str, Any]:
        """Compat: delega para export_epoch_json (formato já é minimalista)."""
        return self.export_epoch_json(epoch=epoch, path=path)


# ---------------------------- helpers externos ----------------------------- #

def convcam_lite(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    out_size: Optional[Union[int, Tuple[int, int]]] = 64,
) -> "ConvCAMAggregator":
    """Factory simples: todas as Conv2d, média incremental, ReLU, normalização, mapa positivo."""
    return ConvCAMAggregator(
        model=model,
        select="conv2d_all",
        out_size=_to_tuple_hw(out_size),
        device=device,
    )


def list_conv2d_layers(model: nn.Module) -> List[str]:
    """Retorna os nomes (qualified) de todas as camadas nn.Conv2d do modelo."""
    return [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]


# ------------------------- compatibilidade de nomes ------------------------- #
class ConvCAMAggregatorRef(ConvCAMAggregator):
    """Alias compatível com versões anteriores."""
    pass
