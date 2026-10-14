"""ResNet-101 feature extraction for ImageNet or two-stage brand checkpoints."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101

from .preprocessing import logo_transform


class ResNet101FeatureExtractor:
    """Return the 2048-D penultimate ResNet-101 representation."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        pretrained: bool = True,
        weights_path: str | Path | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.weights_path = Path(weights_path) if weights_path else None
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained and self.weights_path is None else None
        model = resnet101(weights=weights)
        if pretrained and self.weights_path is not None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Local ResNet-101 weights not found: {self.weights_path}")
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            state = checkpoint
            if isinstance(checkpoint, dict):
                for key in ("model_state", "state_dict", "model"):
                    if isinstance(checkpoint.get(key), dict):
                        state = checkpoint[key]
                        break
            if not isinstance(state, dict):
                raise ValueError(f"No ResNet state_dict found in {self.weights_path}")
            normalized_state = {}
            for key, value in state.items():
                normalized_key = str(key)
                if normalized_key.startswith("module."):
                    normalized_key = normalized_key[7:]
                if isinstance(value, torch.Tensor) and not normalized_key.startswith("fc."):
                    normalized_state[normalized_key] = value
            state = normalized_state
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing_non_head = [key for key in missing if not key.startswith("fc.")]
            if missing_non_head:
                raise RuntimeError(f"ResNet checkpoint does not match standard ResNet-101; missing {missing_non_head[:5]}")
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = logo_transform()
        self.pretrained = pretrained

    @property
    def output_dim(self) -> int:
        return 2048

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(tensor).flatten(1).squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract a batch while preserving the single-image preprocessing."""
        if not images:
            return torch.empty((0, self.output_dim), dtype=torch.float32)
        tensor = torch.stack([self.transform(image) for image in images]).to(self.device)
        return self.model(tensor).flatten(1).cpu()

    @torch.no_grad()
    def extract_path(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.extract(image.convert("RGB"))


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
