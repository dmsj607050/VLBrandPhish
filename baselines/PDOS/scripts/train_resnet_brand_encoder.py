"""Train the standard torchvision ResNet-101 brand encoder in two stages.

Stage 1: supervised Logo 2K+ classification.
Stage 2: initialize from Stage 1 and fine-tune on the 277-brand target list.

The classifier head is used only during training.  The inference pipeline
loads the resulting checkpoint and removes ``fc`` to obtain the 2048-D vector.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet101
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdos.preprocessing import IMAGENET_MEAN, IMAGENET_STD


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_EXCLUDED = {"homepage.png", "loginpage.png", "homepage_html.txt", "login_html.txt", "tfidf.csv"}


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image), label
        except Exception as exc:
            raise RuntimeError(f"Cannot read training image: {path}") from exc


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _split_class_files(class_dirs: list[Path], val_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    names: list[str] = []
    for class_dir in class_dirs:
        files = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            continue
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
        n_val = min(n_val, len(files) - 1) if len(files) > 1 else 0
        cut = len(files) - n_val
        label = len(names)
        names.append(class_dir.name)
        train.extend((path, label) for path in files[:cut])
        val.extend((path, label) for path in files[cut:])
    if not train or not val:
        raise RuntimeError(f"Could not create non-empty train/validation splits under {class_dirs[0].parent if class_dirs else '<empty>'}")
    return train, val, names


def _logo2k_splits(root: Path, val_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Logo 2K+ root not found: {root}")
    # Logo 2K+ is category/brand/image.  A brand is a class, not the outer
    # category, so labels remain stable across category directories.
    brand_dirs = sorted(
        brand_dir
        for category in root.iterdir() if category.is_dir()
        for brand_dir in category.iterdir() if brand_dir.is_dir()
    )
    return _split_class_files(brand_dirs, val_ratio, seed)


def _target_splits(root: Path, val_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"277 target-list root not found: {root}")
    brand_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    # The target-list contains auxiliary html/text and sometimes full-page
    # screenshots.  Match the companion experiment's explicit exclusions.
    samples: list[tuple[Path, int]] = []
    valid_dirs: list[Path] = []
    for brand_dir in brand_dirs:
        files = [path for path in brand_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.name.lower() not in DEFAULT_EXCLUDED]
        if files:
            valid_dirs.append(brand_dir)
    if len(valid_dirs) != 277:
        raise ValueError(f"Expected 277 target-list brand directories with images, found {len(valid_dirs)} under {root}")
    return _split_class_files(valid_dirs, val_ratio, seed)


def _transforms():
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"No state_dict found in {path}")
    normalized = {}
    for key, value in checkpoint.items():
        normalized_key = str(key)
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[7:]
        if isinstance(value, torch.Tensor):
            normalized[normalized_key] = value
    return normalized


def _load_imagenet(model: nn.Module, path: Path) -> None:
    state = {key: value for key, value in _load_checkpoint_state(path).items() if not key.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_head = [key for key in missing if not key.startswith("fc.")]
    if missing_non_head:
        raise RuntimeError(f"ImageNet checkpoint does not match standard ResNet-101; missing {missing_non_head[:5]}")
    print(f"Loaded ImageNet initialization: {path} (unexpected={len(unexpected)})")


def _load_stage1_backbone(model: nn.Module, path: Path) -> None:
    state = _load_checkpoint_state(path)
    backbone = {key: value for key, value in state.items() if not key.startswith("fc.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    missing_non_head = [key for key in missing if not key.startswith("fc.")]
    if missing_non_head:
        raise RuntimeError(f"Stage-1 checkpoint does not match standard ResNet-101; missing {missing_non_head[:5]}")
    print(f"Loaded Stage-1 backbone: {path} (missing_head={len(missing)}, unexpected={len(unexpected)})")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> float:
    top = logits.topk(min(topk, logits.shape[1]), dim=1).indices
    return float((top == labels[:, None]).any(dim=1).float().mean().item())


def _run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, amp: bool, description: str) -> tuple[float, float, float]:
    model.train(train)
    total_loss = total_top1 = total_top5 = total = 0
    iterator = tqdm(loader, desc=description, leave=False)
    for images, labels in iterator:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch = labels.size(0)
        total += batch
        total_loss += float(loss.item()) * batch
        total_top1 += _accuracy(logits.detach(), labels, 1) * batch
        total_top5 += _accuracy(logits.detach(), labels, 5) * batch
    return total_loss / total, total_top1 / total, total_top5 / total


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage standard ResNet-101 brand encoder training")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imagenet-weights", type=Path, default=ROOT / "models" / "resnet101-cd907fc2.pth")
    parser.add_argument("--init-from", type=Path, help="Stage-1 checkpoint for stage2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Image.MAX_IMAGE_PIXELS = None
    device = _resolve_device(args.device)
    if args.stage == "stage1":
        train_samples, val_samples, class_names = _logo2k_splits(args.data_root, args.val_ratio, args.seed)
    else:
        train_samples, val_samples, class_names = _target_splits(args.data_root, args.val_ratio, args.seed)
    print(f"{args.stage}: train={len(train_samples)} val={len(val_samples)} classes={len(class_names)} device={device}")
    train_transform, val_transform = _transforms()
    train_loader = DataLoader(ImageClassificationDataset(train_samples, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ImageClassificationDataset(val_samples, val_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = resnet101(weights=None, num_classes=len(class_names))
    if args.stage == "stage1":
        _load_imagenet(model, args.imagenet_weights)
    else:
        if not args.init_from:
            raise ValueError("--init-from Stage-1 checkpoint is required for stage2")
        _load_stage1_backbone(model, args.init_from)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch = 0
    best_top1 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = _load_checkpoint_state(args.resume)
        model.load_state_dict(state, strict=True)
        if isinstance(checkpoint, dict):
            optimizer.load_state_dict(checkpoint.get("optimizer_state", {})) if checkpoint.get("optimizer_state") else None
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_top1 = float(checkpoint.get("best_val_top1", -1.0))
        print(f"Resuming at epoch {start_epoch}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"stage": args.stage, "architecture": "torchvision_resnet101", "num_classes": len(class_names), "class_names": class_names, "data_root": str(args.data_root.resolve()), "seed": args.seed, "val_ratio": args.val_ratio, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "init_from": str(args.init_from.resolve()) if args.init_from else str(args.imagenet_weights.resolve())}
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_loss, train_top1, train_top5 = _run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, amp, f"train {epoch + 1}/{args.epochs}")
        with torch.no_grad():
            val_loss, val_top1, val_top5 = _run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, amp, f"val {epoch + 1}/{args.epochs}")
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} train@1={train_top1:.4f} val_loss={val_loss:.4f} val@1={val_top1:.4f} val@5={val_top5:.4f} time={time.time() - started:.1f}s", flush=True)
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "best_val_top1": max(best_top1, val_top1), "metadata": metadata}
        torch.save(checkpoint, args.output)
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_path = args.output.with_name("best.pt")
            torch.save(checkpoint, best_path)
            print(f"saved best: {best_path}")
    args.output.with_name("training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete: {args.output}")


if __name__ == "__main__":
    main()
