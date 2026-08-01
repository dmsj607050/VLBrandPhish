"""Load a checkpoint-compatible CLIP or SigLIP encoder for brand matching."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .fusion import LogoTypeRouter, ResidualAdapter


def _is_valid_hf_model_dir(path: str) -> bool:
    root = str(path or "").strip()
    if not root or not os.path.isdir(root):
        return False
    required = ["config.json", "preprocessor_config.json"]
    return all(os.path.isfile(os.path.join(root, name)) for name in required)


def _resolve_siglip_model_dir(model_name: str, *, ckpt_path: str, model_dir: str = "") -> str:
    # Model directory precedence: explicit --model-dir, checkpoint metadata, then a local sibling directory.
    explicit_dir = str(model_dir or "").strip()
    if explicit_dir:
        if not _is_valid_hf_model_dir(explicit_dir):
            raise RuntimeError(
                "--model-dir must contain Hugging Face model files such as config.json and "
                "preprocessor_config.json."
            )
        return os.path.abspath(explicit_dir)

    raw = str(model_name or "").strip()
    if raw and _is_valid_hf_model_dir(raw):
        return os.path.abspath(raw)

    base = os.path.basename(raw.replace("\\", "/").rstrip("/\\"))
    if not base:
        raise ValueError("SigLIP checkpoint is missing model_name and no local model dir could be inferred")

    ckpt_file = Path(os.path.abspath(ckpt_path))
    workspace_root = ckpt_file.anchor or ""
    candidates = [
        ckpt_file.parent / base,
        ckpt_file.parent.parent / base,
        ckpt_file.parent.parent / "siglip-base-patch16-224",
        ckpt_file.parent.parent.parent / "Brand_Identity_SigLIP" / base,
        ckpt_file.parent.parent.parent / "Brand_Identity_TriSig" / base,
    ]
    for cand in candidates:
        cand_str = str(cand)
        if _is_valid_hf_model_dir(cand_str):
            return os.path.abspath(cand_str)

    raise FileNotFoundError(
        "Unable to resolve a local HuggingFace SigLIP model directory for "
        f"model_name='{raw}'. Checked basename='{base}' near checkpoint '{os.path.abspath(ckpt_path)}'."
    )


class CLIPEncoder:
    """Encode logo crops and prompts using the checkpoint-selected CLIP-family backend."""
    def __init__(self, ckpt_path: str, device: str = "cuda:0", model_dir: str = "") -> None:
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.backend = "clip"
        self.model = None
        self.processor = None
        self.transform = None
        self._clip = None
        self.adapter = None
        self.router = None
        self.has_router = False
        self.router_stats_dim = 2
        self.router_out_dim = 3

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CLIP checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")

        siglip_state = ckpt.get("siglip_state", None)
        if siglip_state is not None:
            self._init_siglip_backend(ckpt=ckpt, ckpt_path=ckpt_path, model_dir=model_dir)
            return

        self._init_clip_backend(ckpt=ckpt, ckpt_path=ckpt_path)

    def _init_clip_backend(self, *, ckpt: dict, ckpt_path: str) -> None:
        try:
            import clip  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'clip'. Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self._clip = clip
        self.backend = "clip"
        state = ckpt.get("clip_state", None)
        if state is None:
            state = ckpt.get("model_state", None)
        if state is None:
            raise KeyError(
                "Checkpoint must contain 'clip_state' or 'model_state' for CLIP, "
                "or 'siglip_state' for SigLIP"
            )

        if isinstance(state, dict):
            state = {
                k: v
                for k, v in state.items()
                if (".lora_" not in str(k))
                and (not str(k).endswith("lora_A"))
                and (not str(k).endswith("lora_B"))
            }

        model = clip.model.build_model(state).to(self.device)
        model = model.float().eval()
        self.model = model

        input_size = 224
        if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
            input_size = int(model.visual.input_resolution)

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        with torch.no_grad():
            dummy = torch.zeros((1, 3, input_size, input_size), device=self.device, dtype=torch.float32)
            self.embed_dim = int(self.model.encode_image(dummy).shape[-1])

    def _init_siglip_backend(self, *, ckpt: dict, ckpt_path: str, model_dir: str = "") -> None:
        try:
            from transformers import AutoProcessor, SiglipModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'transformers' for SigLIP checkpoint support."
            ) from e

        model_dir = _resolve_siglip_model_dir(
            str(ckpt.get("model_name") or (ckpt.get("cfg") or {}).get("model_name") or ""),
            ckpt_path=ckpt_path,
            model_dir=model_dir,
        )
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = SiglipModel.from_pretrained(model_dir, local_files_only=True).to(self.device)
        model.load_state_dict(ckpt["siglip_state"], strict=True)
        model = model.float().eval()

        self.backend = "siglip"
        self.processor = processor
        self.model = model

        embed_dim = int(getattr(model.config, "projection_dim", 768))
        cfg = ckpt.get("cfg") or {}
        if "adapter_state" in ckpt:
            adapter = ResidualAdapter(
                embed_dim=embed_dim,
                hidden_dim=int(cfg.get("adapter_hidden_dim", 256)),
                dropout=float(cfg.get("adapter_dropout", 0.1)),
            ).to(self.device)
            adapter.load_state_dict(ckpt["adapter_state"], strict=True)
            self.adapter = adapter.float().eval()
        if "router_state" in ckpt:
            router_state = ckpt["router_state"]
            # Infer router dimensions from the checkpoint so 2- and 3-branch variants remain compatible.
            stats_dim = 2
            out_dim = 3
            try:
                first_weight = router_state.get("net.0.weight", None)
                last_weight = router_state.get("net.4.weight", None)
                if first_weight is not None and int(first_weight.shape[1]) >= int(embed_dim):
                    stats_dim = int(first_weight.shape[1]) - int(embed_dim)
                if last_weight is not None and int(last_weight.shape[0]) > 0:
                    out_dim = int(last_weight.shape[0])
            except Exception:
                stats_dim = 2
                out_dim = 3
            router = LogoTypeRouter(
                embed_dim=embed_dim,
                hidden_dim=int(cfg.get("router_hidden_dim", 256)),
                dropout=float(cfg.get("router_dropout", 0.1)),
                stats_dim=int(stats_dim),
                out_dim=int(out_dim),
            ).to(self.device)
            router.load_state_dict(router_state, strict=True)
            self.router = router.float().eval()
            self.has_router = True
            self.router_stats_dim = int(stats_dim)
            self.router_out_dim = int(out_dim)

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=self.device, dtype=torch.float32)
            self.embed_dim = int(self.model.get_image_features(pixel_values=dummy).shape[-1])

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 64) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)

        out: List[torch.Tensor] = []
        for start in range(0, len(images), int(batch_size)):
            batch_imgs = images[start : start + int(batch_size)]
            if self.backend == "siglip":
                feat = self._encode_siglip_images(batch_imgs, apply_adapter=False)
            else:
                assert self.transform is not None and self.model is not None
                x = torch.stack([self.transform(im.convert("RGB")) for im in batch_imgs], dim=0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(x).float()
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            out.append(feat)
        return torch.cat(out, dim=0)

    def _encode_siglip_images(self, images: Sequence[Image.Image], *, apply_adapter: bool) -> torch.Tensor:
        assert self.processor is not None and self.model is not None
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        x = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(pixel_values=x).float()
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            if apply_adapter and self.adapter is not None:
                feat = F.normalize(feat + self.adapter(feat), dim=-1)
        return feat

    def encode_query_images(self, images: Sequence[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Encode query-side logo crops, applying the residual adapter when available.

        Differs from :meth:`encode_images` in that the adapter (if loaded) is applied
        to query crops so the router sees adapted features consistent with training.
        """
        if not images:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)
        if self.backend != "siglip":
            return self.encode_images(images, batch_size=batch_size)

        out: List[torch.Tensor] = []
        for start in range(0, len(images), int(batch_size)):
            batch_imgs = images[start : start + int(batch_size)]
            out.append(self._encode_siglip_images(batch_imgs, apply_adapter=True))
        return torch.cat(out, dim=0)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)

        out: List[torch.Tensor] = []
        for start in range(0, len(texts), int(batch_size)):
            batch_txt = [str(x) for x in texts[start : start + int(batch_size)]]
            if self.backend == "siglip":
                assert self.processor is not None and self.model is not None
                inputs = self.processor(text=batch_txt, return_tensors="pt", padding=True, truncation=True)
                token_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    feat = self.model.get_text_features(**token_inputs).float()
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                assert self._clip is not None and self.model is not None
                tokens = self._clip.tokenize(batch_txt, truncate=True).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_text(tokens).float()
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            out.append(feat)
        return torch.cat(out, dim=0)

    def route_weights_from_query_features(self, query_features: torch.Tensor) -> torch.Tensor:
        """Produce per-crop branch weights (image/text/ocr) from query features.

        Returns a ``[N, 3]`` tensor. When no router is available, fall back to a
        uniform ``1/3`` distribution. Legacy 2-branch routers are padded into the
        3-column shape expected by downstream consumers.
        """
        n = int(query_features.shape[0]) if query_features.ndim >= 1 else 0
        if n <= 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if self.backend != "siglip" or self.router is None:
            # A model without a trained router uses uniform image/text/OCR branch weights.
            return torch.full((n, 3), 1.0 / 3.0, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = query_features.to(self.device).float()
            if int(getattr(self, "router_out_dim", 3)) == 2:
                # Pad legacy two-branch router outputs to the common three-branch interface.
                stats_dim = int(getattr(self, "router_stats_dim", 6))
                branch_stats = torch.zeros((n, stats_dim), dtype=torch.float32, device=self.device)
                weights2 = self.router(q, branch_stats=branch_stats)
                weights3 = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
                weights3[:, 0:2] = weights2
                return weights3
            # Inference has no OCR branch yet; explicitly mark it unavailable for the 3-branch router.
            ocr_available = torch.zeros((n,), dtype=torch.bool, device=self.device)
            return self.router(q, ocr_available=ocr_available)
