import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union

def generate_architecture_spec_once(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,   # se None, tenta inferir
    file_path: str = "ativacoes.txt",
    max_linear_neurons: Optional[int] = None,        # ex.: 64 para limitar
    default_spatial: int = 224,
) -> str:
    # --- detectar device/dtype do modelo ---
    def _first_tensor(mod: nn.Module):
        for p in mod.parameters(recurse=True): return p
        for b in mod.buffers(recurse=True):    return b
        return None
    ref = _first_tensor(model)
    if ref is None:
        device, dtype = torch.device("cpu"), torch.float32
    else:
        device, dtype = ref.device, ref.dtype
        if device.type == "cpu": dtype = torch.float32  # mais seguro

    # --- inferir input_shape se não vier ---
    def _infer_input_shape(mod: nn.Module):
        for m in mod.modules():
            if isinstance(m, nn.Conv2d):
                return (1, m.in_channels, default_spatial, default_spatial)
            if isinstance(m, nn.Conv1d):
                return (1, m.in_channels, default_spatial)
            if isinstance(m, nn.Conv3d):
                s = default_spatial
                return (1, m.in_channels, s, s, s)
        return None
    if input_shape is None:
        input_shape = _infer_input_shape(model) or (1, 3, default_spatial, default_spatial)

    # --- dummy input compatível ---
    x = torch.zeros(input_shape, device=device, dtype=dtype)

    was_training = model.training
    model.eval()

    entries: List[str] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _append_linear(F: int):
        if F <= 0: F = 1
        if max_linear_neurons is not None:
            F = min(F, max_linear_neurons)
        entries.append("[" + ", ".join("0" for _ in range(F)) + "]")

    # hooks auto-removíveis
    def _hook_conv2d(m: nn.Conv2d):
        h = [None]
        def hook(mod, inp, out):
            if h[0]: h[0].remove()
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                _, C, H, W = out.shape
                # >>> formato {HxWxC} com 'x' <<<
                entries.append(f"{{{H}x{W}x{C}}}")
        h[0] = m.register_forward_hook(hook); handles.append(h[0])

    def _hook_conv1d(m: nn.Conv1d):
        h = [None]
        def hook(mod, inp, out):
            if h[0]: h[0].remove()
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                _, C, L = out.shape
                entries.append(f"{{{L}x1x{C}}}")  # convenção p/ 1D
        h[0] = m.register_forward_hook(hook); handles.append(h[0])

    def _hook_conv3d(m: nn.Conv3d):
        h = [None]
        def hook(mod, inp, out):
            if h[0]: h[0].remove()
            if isinstance(out, torch.Tensor) and out.dim() == 5:
                _, C, D, H, W = out.shape
                entries.append(f"{{{D*H}x{W}x{C}}}")  # compactação simples
        h[0] = m.register_forward_hook(hook); handles.append(h[0])

    def _hook_linear(m: nn.Linear):
        h = [None]
        def hook(mod, inp, out):
            if h[0]: h[0].remove()
            F = int(out.shape[-1]) if isinstance(out, torch.Tensor) else int(m.out_features)
            _append_linear(F)
        h[0] = m.register_forward_hook(hook); handles.append(h[0])

    # registra apenas os tipos desejados
    for m in model.modules():
        if isinstance(m, nn.Conv2d): _hook_conv2d(m)
        elif isinstance(m, nn.Conv1d): _hook_conv1d(m)
        elif isinstance(m, nn.Conv3d): _hook_conv3d(m)
        elif isinstance(m, nn.Linear): _hook_linear(m)

    # único forward, seguro
    try:
        with torch.inference_mode():
            _ = model(x)
    finally:
        for h in handles:
            try: h.remove()
            except: pass
        if was_training: model.train()

    spec_str = "[" + ", ".join(entries) + "]"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(spec_str)

    print(f"Especificação da arquitetura salva em '{file_path}':")
    return spec_str
