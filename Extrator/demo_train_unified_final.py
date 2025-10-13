# demo_train_unified_final.py
import os, json, torch, time, traceback
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# IMPORTANTE: usar a versÃ£o unificada fornecida
from unified_cam_agg_unified import UnifiedCAMAggregator

# ------------------------------------------------------------
# Dataset sintÃ©tico (igual aos demos): quadrado Ã  esquerda/direita
# ------------------------------------------------------------
def make_synthetic_images(n: int, side: int = 28, patch: int = 8, noise: float = 0.05, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(n, 1, side, side)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        cls = i % 2
        y[i] = cls
        img = torch.zeros(1, side, side)
        c0 = 2 if cls == 0 else side - patch - 2
        r0 = (side - patch)//2
        img[0, r0:r0+patch, c0:c0+patch] = 1.0
        img += noise * torch.rand(1, side, side, generator=g)
        X[i] = img.clamp(0,1)
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]

# ------------------------------------------------------------
# Modelo CNN pequeno
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.relu  = nn.ReLU(inplace=False)   # consistente com o ref
        self.fc1   = nn.Linear(16*7*7, 32)    # 28->14->7
        self.fc2   = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)  # 14x14
        x = self.relu(self.conv2(x)); x = self.pool(x)  # 7x7
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------------------------------------
# AvaliaÃ§Ã£o simples
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred==yb).sum().item()
        total   += yb.numel()
    return correct/total if total else 0.0

def main():
    print("CWD:", os.path.abspath(os.getcwd()))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = base_dir + "/logs_unified_final"
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "_write_ok.txt"), "w", encoding="utf-8") as f: f.write("ok\n")

    # Dados
    Xtr, ytr = make_synthetic_images(400, seed=123)
    Xva, yva = make_synthetic_images(100, seed=456)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SmallCNN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    # --- Entradas FIXAS para as ativaÃ§Ãµes/refs (CONV + DENSE) ---
    # pega o primeiro batch e usa somente 10 imagens como referÃªncia fixa
    ref_x = next(iter(train_loader))[0][:10].clone().detach().to(device)
    print("DEBUG ref_x shape:", tuple(ref_x.shape))

    # --- Agregador UNIFICADO (usa os mÃ³dulos unificados de CONV e DENSE) ---
    agg = UnifiedCAMAggregator(
        model,
        device=device,
        conv_out_size=(64,64),          # saÃ­da dos mapas das CONVs (Grad-CAM) em 64x64
        conv_ref_max_channels=8,        # atÃ© 8 canais salvos por camada para as refs
        conv_ref_act_size=32,           # ativaÃ§Ãµes redimensionadas para 32x32 no JSON
        dense_max_features=512,         # corta vetores de entrada muito grandes (opcional)
        run_id="demo_unified_final"
    )

    # ------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------
    EPOCHS = 20  # "nÃºmero maior de Ã©pocas"
    out_path = None
    try:
        for epoch in range(EPOCHS):
            # Inicia Ã©poca: registra hooks DENSE, zera acumuladores CONV e
            # captura as ATIVAÃ‡Ã•ES das ENTRADAS FIXAS (CONV e DENSE) logo aqui.
            agg.begin_epoch(total_examples=len(train_loader.dataset), ref_inputs=ref_x)

            model.train()
            for bidx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                agg.on_batch_begin(batch_size=xb.size(0))

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()

                # Atualiza agregaÃ§Ãµes apÃ³s o backward (CONV usa grads/acts dos hooks)
                agg.on_batch_end()

                opt.step()

            acc = evaluate(model, val_loader, device)
            out_path = agg.end_epoch(save_dir=logs_dir, epoch=epoch)
            print(f"[epoch {epoch:02d}] acc={acc:.3f} | JSON: {os.path.abspath(out_path)}")

        # InspeÃ§Ã£o do Ãºltimo JSON unificado
        if out_path is not None:
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            conv_layers = list(payload["conv"]["layers"].keys())
            dense_layers = list(payload["dense"]["layers"].keys())
            print("\nResumo do Ãºltimo JSON:")
            print("  CONV layers:", conv_layers)
            print("  DENSE layers:", dense_layers)
    except Exception as e:
        print("\nErro durante a execuÃ§Ã£o:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()