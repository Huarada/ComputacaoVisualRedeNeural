# unified_cam_agg_unified.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import os, json, time
import torch
import torch.nn as nn

# IMPORTANTE: use as versões UNIFICADAS fornecidas pelo usuário
from conv_layer_agg_unified import ConvCAMAggregator
from dense_layer_agg_unified import DenseCAMAggregator

class UnifiedCAMAggregator:
    """
    Unificador de CAM para CONV + DENSE trabalhando APENAS com AMOSTRAS FIXAS.

    - CONV  : usa ConvCAMAggregator (unificado). Captura 'acts' das entradas fixas
              via set_ref_inputs(...) e, durante o treino, agrega Grad-CAM positivo,
              normalizado por amostra e médio por época.
    - DENSE : usa DenseCAMAggregator em modo REF (entradas fixas). Salva ref_inputs
              e ref_acts numa passada sem grad no início da época e acumula um
              heatmap 1-D (Gradient × Activation) por época.

    Saída por época (um único JSON):
    {
      "meta": { "epoch": int, "timestamp": int, "run_id": str },
      "conv":  { ...payload do ConvCAMAggregator (com 'acts' das fixas)... },
      "dense": { ...payload do DenseCAMAggregator (ref mode, sem amostras aleatórias)... }
    }
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device | str] = None,
        *,
        # CONV (visualização dos mapas)
        conv_out_size: Optional[Tuple[int, int] | int] = 64,
        conv_ref_max_channels: int = 8,
        conv_ref_act_size: Tuple[int, int] | int = 32,
        # DENSE (limite opcional de features por camada)
        dense_max_features: Optional[int] = None,
        run_id: str = "default",
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.run_id = str(run_id)

        # --- CONV: versão unificada com captura de ATIVAÇÕES fixas por época
        self.conv = ConvCAMAggregator(
            model,
            select="conv2d_all",
            out_size=conv_out_size,
            device=self.device,
        )
        self._conv_ref_max_channels = int(conv_ref_max_channels)
        self._conv_ref_act_size = conv_ref_act_size

        # --- DENSE: **somente** modo REF (entradas fixas)
        self.dense = DenseCAMAggregator.ref(
            model,
            layer_filter=None,                # monitora todos os nn.Linear
            device=self.device,
            max_features=dense_max_features,  # opcional: cortar vetores muito grandes
        )

        # controle de ciclo
        self._batch_index = 0
        self._batch_size = 0
        self._ref_inputs: Optional[torch.Tensor] = None

    # ----------------- ciclo de época ----------------- #
    def begin_epoch(self, total_examples: int, ref_inputs: Optional[torch.Tensor] = None) -> None:
        """
        ref_inputs: tensor (B*, C, H, W) **fixo** que será usado para:
          - CONV: capturar 'acts' logo no começo da época;
          - DENSE: capturar ref_inputs/ref_acts logo no começo e acumular heatmap.
        """
        # define/atualiza conjunto fixo
        if ref_inputs is not None:
            self._ref_inputs = ref_inputs.detach().to(self.device)
            # DENSE (ref): limite opcional de amostras/fixas (default 10)
            self.dense.set_ref_inputs(self._ref_inputs, max_samples=10)
            # CONV: também define as mesmas referências para capturar ATIVAÇÕES
            self.conv.set_ref_inputs(
                self._ref_inputs,
                max_images=min(10, self._ref_inputs.size(0)),
                max_channels=self._conv_ref_max_channels,
                act_size=self._conv_ref_act_size,
            )

        # inicia denso (registra hooks de backward) e já captura ref_inputs/ref_acts
        self.dense.begin_epoch(total_examples=total_examples)

        # inicia conv (zera acumuladores e CAPTURA 'acts' das refs internamente)
        self.conv.begin_epoch()

        self._batch_index = 0
        self._batch_size = 0

    def on_batch_begin(self, batch_size: int) -> None:
        self._batch_size = int(batch_size)
        # (modo REF não precisa de janela/posições; mantido para compat)
        # se quiser, você pode chamar: self.dense.on_batch_begin(...), mas no REF é no-op

    def on_batch_end(self) -> None:
        """
        Chame após loss.backward(). A CONV usa os hooks de grad/act para
        atualizar a média incremental dos mapas por camada.
        """
        self.conv.update_from_hooks()
        self._batch_index += 1

    def end_epoch(self, save_dir: str, epoch: int) -> str:
        os.makedirs(save_dir, exist_ok=True)

        # DENSE (ref) salva seu JSON próprio; carregamos para unificar
        dense_path = self.dense.end_epoch(save_dir=save_dir, epoch=epoch, run_id=self.run_id)
        with open(dense_path, "r", encoding="utf-8") as f:
            dense_payload = json.load(f)

        # CONV retorna dict já contendo 'acts' das amostras fixas
        conv_payload = self.conv.export_epoch_json(epoch=epoch, path=None)

        final_payload: Dict[str, Any] = {
            "meta": {"epoch": int(epoch), "timestamp": int(time.time()), "run_id": self.run_id},
            "conv": conv_payload,
            "dense": dense_payload,
        }

        out_path = os.path.join(save_dir, f"unified_epoch_{epoch:04d}_{self.run_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)
        return out_path
