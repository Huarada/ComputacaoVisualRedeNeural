import os, json, base64, io, time, traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from PIL import Image
from gen_architecture import generate_architecture_spec_once


from unified_cam_agg_unified import UnifiedCAMAggregator

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

def get_catsdogs_loader(data_root="./data/oxford_pets_auto", batch_size=64):
    tfm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cat_breeds = {
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
        "Siamese", "Sphynx",
    }

    class OxfordPetsBinary(torch.utils.data.Dataset):
        def __init__(self, root, split: str, transform=None):
            self.ds = OxfordIIITPet(root=root, split=split, target_types="category", download=True)
            self.transform = transform
            if hasattr(self.ds, "classes"):
                self._breed_names = list(self.ds.classes)
            else:
                inv = sorted(self.ds.class_to_idx.items(), key=lambda kv: kv[1])
                self._breed_names = [k for k, _ in inv]
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, target = self.ds[idx]
            if isinstance(target, (tuple, list)):
                breed_idx = int(target[0])
            else:
                breed_idx = int(target)
            breed_name = self._breed_names[breed_idx]
            label = 0 if breed_name in cat_breeds else 1
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

    train_ds = OxfordPetsBinary(root=data_root, split="trainval", transform=tfm)
    val_ds   = OxfordPetsBinary(root=data_root, split="test",     transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(128, batch_size), shuffle=False)
    return {"train": train_loader, "val": val_loader}

def fetch_ref_by_indices(dataset, indices, device):
    xs = []
    for i in indices:
        x, _ = dataset[int(i)]
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        xs.append(x)
    ref_x = torch.stack(xs, dim=0).to(device)
    return ref_x

def save_reference_original_pngs(dataset, indices, logs_dir: str, prefix: str = "refs_orig", grid_thumb_max=256):
    os.makedirs(logs_dir, exist_ok=True)
    imgs_pil = []
    for i in indices:
        img_pil, _ = dataset.ds[int(i)]
        if img_pil.mode not in ("RGB", "L"):
            img_pil = img_pil.convert("RGB")
        imgs_pil.append(img_pil)

    for k, im in enumerate(imgs_pil):
        outp = os.path.join(logs_dir, f"{prefix}_{k:03d}.png")
        im.save(outp)

    import math
    n = len(imgs_pil)
    if n == 0:
        return None
    nrow = int(math.sqrt(n))
    if nrow * nrow < n:
        nrow = max(1, nrow)
    ncol = math.ceil(n / nrow)

    thumbs = []
    cell_w = cell_h = 0
    for im in imgs_pil:
        imc = im.copy()
        imc.thumbnail((grid_thumb_max, grid_thumb_max))
        cell_w = max(cell_w, imc.width)
        cell_h = max(cell_h, imc.height)
        thumbs.append(imc)

    grid_w = ncol * cell_w
    grid_h = nrow * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    for idx, imc in enumerate(thumbs):
        r = idx // ncol
        c = idx % ncol
        x = c * cell_w + (cell_w - imc.width) // 2
        y = r * cell_h + (cell_h - imc.height) // 2
        grid.paste(imc, (x, y))

    grid_path = os.path.join(logs_dir, f"{prefix}_grid.png")
    grid.save(grid_path)
    return grid_path

def _plot_acc_then_png(acc_hist, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure()
    xs = list(range(1, len(acc_hist)+1))
    plt.plot(xs, acc_hist)
    plt.scatter([xs[-1]], [acc_hist[-1]])
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (val)")
    plt.grid(True, alpha=0.3)
    png_path = os.path.join(save_dir, f"acc_plot_epoch_{epoch:02d}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path

def train_with_agg(
    model,
    opt,
    crit,
    loader,
    *,
    device="auto",
    epochs=20,
    logs_dir=None,
    run_id="demo_unified_final",
    ref_indices=None,
):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)

    if isinstance(loader, (list, tuple)) and len(loader) == 2:
        train_loader, val_loader = loader[0], loader[1]
    elif isinstance(loader, dict) and "train" in loader and "val" in loader:
        train_loader, val_loader = loader["train"], loader["val"]
    else:
        ds = loader.dataset
        n = len(ds)
        idx = torch.randperm(n)
        n_train = int(0.8*n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=loader.batch_size or 64, shuffle=True)
        val_loader   = DataLoader(Subset(ds, val_idx.tolist()),   batch_size=max(128, (loader.batch_size or 64)), shuffle=False)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = logs_dir or (base_dir + "/logs_unified_final")
    os.makedirs(logs_dir, exist_ok=True)

    if ref_indices is not None and len(ref_indices) > 0:
        ref_x = fetch_ref_by_indices(train_loader.dataset, ref_indices, device)
    else:
        ref_x = next(iter(train_loader))[0][:10].clone().detach().to(device)
    print("DEBUG ref_x shape:", tuple(ref_x.shape))

    refs_orig_png_grid = None
    if ref_indices is not None and len(ref_indices) > 0:
        refs_orig_png_grid = save_reference_original_pngs(train_loader.dataset, ref_indices, logs_dir, prefix="refs_orig")

    agg = UnifiedCAMAggregator(
        model,
        device=device,
        conv_out_size=(64,64),
        conv_ref_max_channels=32,
        conv_ref_act_size=32,
        dense_max_features=64,
        run_id=run_id
    )

    acc_hist = []
    out_path = None
    try:
        for epoch in range(epochs):
            agg.begin_epoch(total_examples=len(train_loader.dataset), ref_inputs=ref_x)

            model.train()
            for bidx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                agg.on_batch_begin(batch_size=xb.size(0))
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                agg.on_batch_end()
                opt.step()

            acc = evaluate(model, val_loader, device)
            acc_hist.append(float(acc))

            out_path = agg.end_epoch(save_dir=logs_dir, epoch=epoch)

            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                png_path = _plot_acc_then_png(acc_hist, logs_dir, epoch)
                payload.setdefault("metrics", {})
                payload["metrics"].update({
                    "acc_per_epoch": acc_hist,
                    "acc_last": acc_hist[-1],
                    # gráfico salvo em arquivo, não mais embutido no JSON
                    "refs_orig_png_grid": os.path.relpath(refs_orig_png_grid, start=logs_dir) if refs_orig_png_grid else None,
                    "ref_indices": list(map(int, ref_indices)) if ref_indices is not None else None,
                })
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                traceback.print_exc()

            print(f"[epoch {epoch:02d}] acc={acc:.3f} | JSON: {os.path.abspath(out_path)}")

        if out_path is not None:
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            conv_layers = list(payload["conv"]["layers"].keys())
            dense_layers = list(payload["dense"]["layers"].keys())
            print("\nResumo do último JSON:")
            print("  CONV layers:", conv_layers)
            print("  DENSE layers:", dense_layers)
    except Exception as e:
        print("\nErro durante a execução:", e)
        traceback.print_exc()

    return out_path


def main():
    loaders = get_catsdogs_loader(data_root="./data/oxford_pets_auto", batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SmallCNN(in_ch=1, num_classes=2).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()
    ref_indices = torch.randint(low=0, high=1001, size=(10,))
    generate_architecture_spec_once(model, input_shape=(64,1,28,28), file_path="ativacoes.txt", max_linear_neurons=64)
    train_with_agg(
        model, opt, crit, loaders,
        device=device, epochs=20,
        run_id="demo_unified_final",
        ref_indices=ref_indices,
    )

if __name__ == "__main__":
    main()
