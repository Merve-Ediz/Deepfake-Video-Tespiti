# ------------------------------------------------------------
# Minimal-but-solid training script for FaceForensics++ (preprocessed RGB frames)
# - Reads hyperparameters from a YAML file (e.g., train/xception.yaml)
# - Expects dataset at: ./datasets/rgb/FaceForensics++/
# - Uses train/val/test json lists if present; otherwise creates a deterministic split
# - Trains a binary classifier (real vs fake) and reports AUC / Accuracy
# ------------------------------------------------------------

import os
import json
import time
import random
import argparse #Komut satırından argüman almak için.
from dataclasses import dataclass # “Sınıf gibi ama çok pratik” veri tutucu yapılar oluşturur.
from typing import List, Tuple, Dict, Optional # Tip ipuçları: kodu okurken anlaşılır kılar.

import numpy as np

import torch
import torch.nn as nn # Neural network katmanları ve loss fonksiyonları.
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

# timm = bir sürü hazır model/backbone barındıran kütüphane.
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int = 1024) -> None: # Amaç: eğitimdeki rastgeleliği kontrol altına almak.
    random.seed(seed)
    np.random.seed(seed) #NumPy rastgeleliğini sabitler.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # “tam deterministik olmasın, hız biraz daha iyi olabilir” demek.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True #input boyutları sabitse hız artırır.


def load_yaml(path: str) -> Dict: #Dict = Sözlük
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None: # “Bu klasör yoksa oluştur” fonksiyonu.
    os.makedirs(path, exist_ok=True)


def list_ids_from_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # IDs can be numbers or strings; normalize to string
    return [str(x) for x in data]


def deterministic_split(ids: List[str], seed: int = 1024, val_ratio: float = 0.1, test_ratio: float = 0.1): # Train/val/test split yapar. val_ratio ve test_ratio default %10.
    rng = random.Random(seed)
    ids = ids.copy() # Orijinal listeyi bozmaz.
    rng.shuffle(ids) # Listeyi karıştırır
    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


# -------------------------
# Dataset
# -------------------------

@dataclass
class FFPaths:
    root: str  # ./datasets/rgb/FaceForensics++
    compression: str  # c23
    # Real folders 
    real_actors: str
    # Fake folders (subset of manipulations)
    fake_dirs: List[str]

    @staticmethod
    def from_root(root: str, compression: str) -> "FFPaths":
        # Only actors real subset
        real_actors = os.path.join(root, "original_sequences", "actors", compression)

        manip_root = os.path.join(root, "manipulated_sequences")
        fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]
        fake_dirs = []
        for m in fake_methods:
            p = os.path.join(manip_root, m, compression)
            if os.path.isdir(p): # Bu klasör gerçekten var mı?
                fake_dirs.append(p)

        return FFPaths(root=root, compression=compression, real_actors=real_actors, fake_dirs=fake_dirs)


def _safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]) # os.listdir(path) = klasör içindeki isimler. [d for d in ... if ...] = sadece klasör olanları al. sorted(...) = alfabetik sırala.


def _collect_frame_paths(video_dir: str) -> List[str]:
    """
    Many preprocessed packs store frames as:
      <video_dir>/frames/<video_id>/000.png ...
    Some store directly images under the video directory.
    """
    frames_dir = os.path.join(video_dir, "frames")
    base = frames_dir if os.path.isdir(frames_dir) else video_dir

    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(exts)] # base içindeki dosyalardan uzantısı uygun olanları al. f.lower() = uzantı büyük küçük harf farkını engelle.
    files.sort() # Dosyaları sıralar.
    return files


class FaceForensicsFramesDataset(Dataset): # Bir “örnek” olarak tek frame döndürmek.
    """
    Builds samples as (frame_tensor, label).
    For each video ID, we sample 'frame_num' frames per epoch (with replacement if needed).
    """
    def __init__( # Python’da sınıfın yapıcı fonksiyonu. Dataset oluşturulurken çağrılır.
        self, # Sınıfın kendisi.
        ff: FFPaths,
        video_ids: List[str],
        frame_num: int,
        resolution: int,
        mean: List[float],
        std: List[float],
        augment: bool = True,
        mode: str = "train",
        max_videos: Optional[int] = None, # Sınırlama yok.
    ):
        super().__init__() # parent class (Dataset) init’ini çağırır.
        self.ff = ff
        self.video_ids = video_ids[:max_videos] if max_videos else video_ids
        self.frame_num = frame_num
        self.mode = mode

        # Transforms
        aug = []
        if augment and mode == "train":
            aug += [transforms.RandomHorizontalFlip(p=0.5)] # %50 ihtimalle görüntüyü yatay çevir.

        self.tf = transforms.Compose(
            aug
            + [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # Dataset’in gerçek “örnek kaynak listesi” kuruluyor.
        self.video_entries: List[Tuple[str, int]] = []

        # Real: ONLY actors
        for vid in self.video_ids:
            p = os.path.join(ff.real_actors, "frames", vid)
            if os.path.isdir(p):
                self.video_entries.append((p, 0))  # 0=real

        # Fake: include all available fake dirs
        for fake_base in ff.fake_dirs:
            for vid in self.video_ids:
                p = os.path.join(fake_base, "frames", vid)
                if os.path.isdir(p):
                    self.video_entries.append((p, 1))  # 1=fake

        if len(self.video_entries) == 0:
            raise FileNotFoundError(
                "No video entries found. Check dataset layout.\n"
                f"Expected real dir like:\n  {ff.real_actors}\n"
                f"Expected fake dirs under manipulated_sequences/*/{ff.compression}\n"
                f"Dataset root provided: {ff.root}"
            )

        # Total length is: num_videos * frame_num
        self._len = len(self.video_entries) * self.frame_num # Bu gerçek frame sayısı değil, “örnekleme sayısı”.

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int): #DataLoader bir örnek istediğinde bu çağrılır.
        ve_idx = idx // self.frame_num
        video_dir, label = self.video_entries[ve_idx]

        frames = _collect_frame_paths(video_dir)

        if len(frames) > 0:
            if self.mode == "train":
                fp = random.choice(frames)
            else:
                fp = frames[idx % len(frames)] #idx % len(frames) ile index taşarsa başa sar. Böylece her evaluation tekrarında aynı sırada frame seçilir (daha adil)
        else:
            # “dummy” siyah görüntü oluştur.
            img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
            x = self.tf(img)
            y = torch.tensor(label, dtype=torch.long)
            return x, y

        img = Image.open(fp).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -------------------------
# Model
# -------------------------

def build_model(model_name: str, num_classes: int = 2):
    """
    Uses timm if available (preferred for xception). Falls back to resnet18.
    """
    name = (model_name or "").lower()
    if _HAS_TIMM:
        for cand in [name, "xception", "xception41", "xception65"]:
            try:
                m = timm.create_model(cand, pretrained=True, num_classes=num_classes)
                return m, cand
            except Exception:
                pass

    from torchvision.models import resnet18
    m = resnet18(weights="DEFAULT")
    m.fc = nn.Linear(m.fc.in_features, num_classes) # ResNet’in son fully-connected katmanını 2 sınıfa göre değiştir. m.fc.in_features: önceki katmanın çıkış boyutu. nn.Linear(in, out): düz lineer katman.
    return m, "resnet18_fallback"


# -------------------------
# Train / Eval
# -------------------------

@torch.no_grad() # Bu fonksiyon içinde gradient hesaplanmasın.
def evaluate(model, loader, device) -> Dict[str, float]: #Modeli val/test set üzerinde ölçer.
    model.eval() # Modeli evaluation moduna alır.
    ys, ps = [], [] # Gerçek etiketler (ys) ve olasılıklar (ps) listeleri.
    preds = [] # Tahmin edilen sınıflar listesi.
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x) # Modelin ham çıktısı: logits.
        prob = torch.softmax(logits, dim=1)[:, 1]  # P(fake) Sınıf boyutunda (2 sınıf) olasılığa çevirir.
        pred = torch.argmax(logits, dim=1)

        ps.append(prob.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int32) # ys listelerini tek dizi yap.
    y_prob = np.concatenate(ps).astype(np.float32)
    y_pred = np.concatenate(preds).astype(np.int32)

    auc = float("nan")
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob)
    except Exception:
        pass

    acc = accuracy_score(y_true, y_pred)
    return {"auc": float(auc), "acc": float(acc)}


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0 # Toplam loss biriktirici.
    n = 0 # Toplam örnek sayısı.
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0) # bu batch’te kaç örnek var?
        running += loss.item() * bs # loss.item() = tensor içindeki sayıyı python float’a çevir. * bs: batch ortalaması yerine toplam kayıp gibi biriktirmek için. sonra toplam örnek sayısına bölecek.
        n += bs # örnek sayısını topla.
    return running / max(1, n) # max(1,n): n=0 gibi saçma durumda 0’a bölme olmasın.


# -------------------------
# Main
# -------------------------

def parse_args(): # Komut satırı parametrelerini ayarlayan fonksiyon.
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", type=str, default="./train/xception.yaml", help="path to detector YAML file")
    ap.add_argument("--data_root", type=str, default="./datasets/rgb/FaceForensics++", help="FF++ preprocessed root")
    ap.add_argument("--max_videos", type=int, default=0, help="debug: limit number of video IDs (0=no limit)")
    ap.add_argument("--no_aug", action="store_true", help="disable augmentation") # augmentation kapatmak için.
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.detector_path)

    seed = int(cfg.get("manualSeed", 1024))
    set_seed(seed)

    compression = cfg.get("compression", "c23")
    train_bs = int(cfg.get("train_batchSize", 32))
    test_bs = int(cfg.get("test_batchSize", 32))
    workers = int(cfg.get("workers", 4))
    n_epochs = int(cfg.get("nEpochs", 10))
    frame_num = cfg.get("frame_num", {"train": 32, "test": 32})
    frame_train = int(frame_num.get("train", 32))
    frame_test = int(frame_num.get("test", 32))
    resolution = int(cfg.get("resolution", 256))
    mean = cfg.get("mean", [0.5, 0.5, 0.5])
    std = cfg.get("std", [0.5, 0.5, 0.5])

    model_name = cfg.get("model_name", "xception")
    metric_scoring = cfg.get("metric_scoring", "auc").lower()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("cuda", True) else "cpu")

    # Split IDs: prefer provided JSONs
    json_candidates = [
        os.path.join(args.data_root, "train.json"),
        os.path.join(args.data_root, "val.json"),
        os.path.join(args.data_root, "test.json"),
        os.path.join(os.path.dirname(args.data_root), "train.json"), # bir üst klasörde train.json olma ihtimali
        os.path.join(os.path.dirname(args.data_root), "val.json"),
        os.path.join(os.path.dirname(args.data_root), "test.json"),
    ]
    train_json = next((p for p in json_candidates if p.endswith("train.json") and os.path.isfile(p)), None) #generator’dan ilk uygun elemanı al.
    val_json = next((p for p in json_candidates if p.endswith("val.json") and os.path.isfile(p)), None)
    test_json = next((p for p in json_candidates if p.endswith("test.json") and os.path.isfile(p)), None)

    ff = FFPaths.from_root(args.data_root, compression=compression) # Dataset yol objesini oluştur. ff.real_actors, ff.fake_dirs gibi hazır olacak.

    if train_json and val_json and test_json:
        train_ids = list_ids_from_json(train_json)
        val_ids = list_ids_from_json(val_json)
        test_ids = list_ids_from_json(test_json)
        split_note = "Using provided JSON splits."
    else:
        # fallback split: infer IDs from actors / JSON split yoksa fallback yap. (Yedek fonksiyon)
        ids = _safe_listdir(ff.real_actors)
        if len(ids) == 0:
            for d in ff.fake_dirs:
                ids = _safe_listdir(d)
                if ids:
                    break
        if len(ids) == 0:
            raise FileNotFoundError("Could not infer any video IDs from dataset folders.")

        train_ids, val_ids, test_ids = deterministic_split(ids, seed=seed, val_ratio=0.1, test_ratio=0.1) # deterministik split: 80/10/10
        split_note = "JSON splits not found; using deterministic 80/10/10 split."

    if args.max_videos and args.max_videos > 0: # max_videos verildiyse ve 0’dan büyükse…
        train_ids = train_ids[:args.max_videos]
        val_ids = val_ids[: max(1, args.max_videos // 5)] # val’i train’in yaklaşık 1/5’i kadar tut.
        test_ids = test_ids[: max(1, args.max_videos // 5)]

    print("--------------------------------------------------")
    print(f"Config: {args.detector_path}")
    print(f"Dataset root: {args.data_root}")
    print(f"Compression: {compression}")
    print(f"Split: {split_note}")
    print(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)} | Test IDs: {len(test_ids)}")
    print(f"Frames/video (train/test): {frame_train}/{frame_test} | Resolution: {resolution}")
    print(f"Batch (train/test): {train_bs}/{test_bs} | Workers: {workers}")
    print(f"Device: {device}")
    print("--------------------------------------------------")

    use_aug = bool(cfg.get("use_data_augmentation", True)) and (not args.no_aug) # Augmentation seçimi YAML use_data_augmentation true olmalı

    train_ds = FaceForensicsFramesDataset(
        ff=ff, video_ids=train_ids, frame_num=frame_train, resolution=resolution,
        mean=mean, std=std, augment=use_aug, mode="train",
        max_videos=None
    )
    val_ds = FaceForensicsFramesDataset(
        ff=ff, video_ids=val_ids, frame_num=frame_test, resolution=resolution,
        mean=mean, std=std, augment=False, mode="val",
        max_videos=None
    )
    test_ds = FaceForensicsFramesDataset(
        ff=ff, video_ids=test_ids, frame_num=frame_test, resolution=resolution,
        mean=mean, std=std, augment=False, mode="test",
        max_videos=None
    )

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True) # shuffle=True: her epoch’ta örnekleri karıştır drop_last=True: son batch küçük kalırsa at. BatchNorm gibi katmanlarda stabilite için tercih edilir
    val_loader = DataLoader(val_ds, batch_size=test_bs, shuffle=False, num_workers=max(0, workers // 2), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=test_bs, shuffle=False, num_workers=max(0, workers // 2), pin_memory=True)

    model, used_name = build_model(model_name=model_name, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # Loss fonksiyonu

    opt_cfg = cfg.get("optimizer", {})
    opt_type = (opt_cfg.get("type", "adam") or "adam").lower()
    lr = float(opt_cfg.get("adam", {}).get("lr", 2e-4)) if opt_type == "adam" else float(opt_cfg.get("sgd", {}).get("lr", 2e-4))
    wd = float(opt_cfg.get("adam", {}).get("weight_decay", 5e-4)) if opt_type == "adam" else float(opt_cfg.get("sgd", {}).get("weight_decay", 5e-4))

    if opt_type == "sgd":
        momentum = float(opt_cfg.get("sgd", {}).get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        beta1 = float(opt_cfg.get("adam", {}).get("beta1", 0.9))
        beta2 = float(opt_cfg.get("adam", {}).get("beta2", 0.999))
        eps = float(opt_cfg.get("adam", {}).get("eps", 1e-8))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

    log_dir = cfg.get("log_dir", "./logs")
    ensure_dir(log_dir) # yoksa oluştur.
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    best_key = "auc" if metric_scoring == "auc" else "acc"
    best_score = -1.0 # AUC/acc 0-1 arası olduğundan -1 kesin daha kötü; ilk iyi skor bunu geçer.

    print(f"Model: {used_name} (requested: {model_name})")
    print(f"Optimizer: {opt_type} | lr={lr} | weight_decay={wd}")
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device) # validation ölç.

        score = val_metrics.get(best_key, float("nan"))
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{n_epochs:02d}] "
            f"loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"time={elapsed:.1f}s"
        )

        if not np.isnan(score) and score > best_score:
            best_score = score # score NaN değilse ve önceki best’ten büyükse best’i güncelle.
            ckpt_path = os.path.join(ckpt_dir, f"best_{used_name}_{compression}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"  -> Saved best checkpoint to: {ckpt_path}")

    best_ckpt = os.path.join(ckpt_dir, f"best_{used_name}_{compression}.pth")
    if os.path.isfile(best_ckpt):
        state = torch.load(best_ckpt, map_location=device) # checkpoint’i yükle.
        model.load_state_dict(state["model"], strict=False) # ağırlıkları modele yükle. strict=False: birebir tüm katman isimleri uyuşmazsa bile “uyuşanları yükle”. Bazı model sürümleri/başlıkları fark ederse patlamasın diye daha toleranslı
        print(f"Loaded best checkpoint for test: {best_ckpt}")

    test_metrics = evaluate(model, test_loader, device)
    print("--------------------------------------------------")
    print(f"TEST  acc={test_metrics['acc']:.4f}  auc={test_metrics['auc']:.4f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()