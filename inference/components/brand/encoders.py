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
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        model_dir: str = "",
        skip_text_encoder: bool = False,
    ) -> None:
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.backend = "clip"
        self.model = None
        self.processor = None
        self.transform = None
        self._clip = None
        self._ckpt_label_to_name = {}
        self._ckpt_text_prototypes = None
        self._ckpt_image_prototypes = None
        self._skip_text_encoder = bool(skip_text_encoder)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CLIP checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")

        # Capture prototypes + label→name mapping so brand_text_mode=ckpt_text_proto_mean can
        # reuse the pre-encoded text prototypes from training (avoids loading the tokenizer,
        # which crashes under Windows sentencepiece protobuf for some SIGLIP checkpoints).
        l2n = ckpt.get("label_to_name")
        if isinstance(l2n, dict) and l2n:
            self._ckpt_label_to_name = {int(k): str(v) for k, v in l2n.items()}
        tp = ckpt.get("text_prototypes")
        if tp is not None:
            self._ckpt_text_prototypes = torch.as_tensor(tp, dtype=torch.float32, device=self.device).detach()
            if self._ckpt_text_prototypes.ndim == 2:
                norms = self._ckpt_text_prototypes.norm(dim=-1, keepdim=True)
                self._ckpt_text_prototypes = self._ckpt_text_prototypes / (norms + 1e-6)
        ip = ckpt.get("image_prototypes")
        if ip is not None:
            self._ckpt_image_prototypes = torch.as_tensor(ip, dtype=torch.float32, device=self.device).detach()

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
            from transformers import AutoProcessor, SiglipModel, SiglipImageProcessor
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'transformers' for SigLIP checkpoint support."
            ) from e

        model_dir = _resolve_siglip_model_dir(
            str(ckpt.get("model_name") or (ckpt.get("cfg") or {}).get("model_name") or ""),
            ckpt_path=ckpt_path,
            model_dir=model_dir,
        )

        model = SiglipModel.from_pretrained(model_dir, local_files_only=True).to(self.device)
        model.load_state_dict(ckpt["siglip_state"], strict=True)
        model = model.float().eval()

        self.backend = "siglip"
        self.model = model

        # Try to load the full processor (image + tokenizer). This requires a
        # working sentencepiece installation. If it fails (e.g. Windows
        # sentencepiece 0.2.1 crash), fall back to image-only processor.
        try:
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            self._processor_is_image_only = False
        except Exception:
            # Fall back to image-only processor
            img_proc_dir = str(model_dir)
            try:
                img_proc = SiglipImageProcessor.from_pretrained(img_proc_dir, local_files_only=True)
            except Exception:
                img_proc = SiglipImageProcessor(
                    do_resize=True,
                    size={"height": 224, "width": 224},
                    resample=3,  # BICUBIC
                    do_rescale=True,
                    rescale_factor=1.0 / 255.0,
                    do_normalize=True,
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                    do_convert_rgb=True,
                )
            self.processor = img_proc
            self._processor_is_image_only = True

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=self.device, dtype=torch.float32)
            self.embed_dim = int(self.model.get_image_features(pixel_values=dummy).shape[-1])

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Encode reference logo images into normalized embedding vectors."""
        if not images:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)

        out: List[torch.Tensor] = []
        for start in range(0, len(images), int(batch_size)):
            batch_imgs = images[start : start + int(batch_size)]
            if self.backend == "siglip":
                feat = self._encode_siglip_images(batch_imgs)
            else:
                assert self.transform is not None and self.model is not None
                x = torch.stack([self.transform(im.convert("RGB")) for im in batch_imgs], dim=0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(x).float()
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            out.append(feat)
        return torch.cat(out, dim=0)

    def _encode_siglip_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        assert self.processor is not None and self.model is not None
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        x = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(pixel_values=x).float()
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat


    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
        """Encode brand prompt texts into normalized embedding vectors."""
        if not texts:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)

        if self.backend == "siglip" and bool(getattr(self, "_processor_is_image_only", False)):
            raise RuntimeError(
                "SigLIP text encoding is disabled because the tokenizer (sentencepiece) was "
                "skipped to avoid Windows protobuf crashes. Use brand_text_mode that does not "
                "require text encoding, e.g. ckpt_text_proto_mean, or rebuild this encoder "
                "with skip_text_encoder=False after fixing the sentencepiece environment."
            )

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
