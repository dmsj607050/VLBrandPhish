from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProtoCache:
    prototypes: torch.Tensor
    label_to_prompts: Dict[int, List[str]]
    labels: Optional[List[int]] = None
    label_to_index: Optional[Dict[int, int]] = None


def _set_or_register_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Update a registered buffer in-place, or register it if it doesn't exist yet."""
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor)


def _resolve_embed_dim(siglip_model) -> int:
    config = siglip_model.config
    if hasattr(config, "projection_dim"):
        return int(config.projection_dim)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
        return int(config.vision_config.hidden_size)
    raise AttributeError("Unable to resolve SigLIP embedding dimension.")


@torch.no_grad()
def build_text_prototypes(
    siglip_model,
    tokenizer,
    label_to_prompts: Dict[int, List[str]],
    device: torch.device,
    *,
    prompt_ensemble: str = "mean",
    batch_size: int = 64,
) -> ProtoCache:
    siglip_model.eval()
    labels = sorted(label_to_prompts.keys())
    prototypes: List[torch.Tensor] = []

    for label in labels:
        prompts = label_to_prompts[label]
        features: List[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            tokens = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            text_features = siglip_model.get_text_features(**tokens).float()
            features.append(text_features)

        prompt_features = torch.cat(features, dim=0)
        if str(prompt_ensemble).lower() == "max":
            idx = prompt_features.norm(dim=-1).argmax()
            proto = prompt_features[idx]
        else:
            proto = prompt_features.mean(dim=0)
        prototypes.append(F.normalize(proto, dim=-1))

    stacked = torch.stack(prototypes, dim=0)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    return ProtoCache(
        prototypes=stacked,
        label_to_prompts=label_to_prompts,
        labels=labels,
        label_to_index=label_to_index,
    )


class SigLIPProtoModel(nn.Module):
    def __init__(self, siglip_model):
        super().__init__()
        self.siglip = siglip_model
        _set_or_register_buffer(self, "text_prototypes", torch.empty(0))

    @property
    def embed_dim(self) -> int:
        return _resolve_embed_dim(self.siglip)

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        _set_or_register_buffer(self, "text_prototypes", F.normalize(prototypes.float(), dim=-1))

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.siglip.get_image_features(pixel_values=images).float()
        return F.normalize(features, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.siglip.logit_scale.exp().clamp(1.0, 100.0)

    def get_logit_bias(self) -> torch.Tensor:
        return self.siglip.logit_bias.clamp(-10.0, 10.0)

    def forward_logits_from_features(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("Text prototypes not set.")
        prototypes = F.normalize(self.text_prototypes, dim=-1)
        return self.get_logit_scale() * (image_features @ prototypes.t())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image_features(images)
        return self.forward_logits_from_features(image_features)


class SigLIPTriSigModel(nn.Module):
    def __init__(
        self,
        siglip_model,
        *,
        image_proto_agg: str = "attention",
        image_proto_agg_tau: float = 0.07,
        image_weight: float = 0.8,
        text_weight: float = 0.2,
    ):
        super().__init__()
        self.siglip = siglip_model
        self.image_proto_agg = str(image_proto_agg).lower()
        self.image_proto_agg_tau = float(image_proto_agg_tau)
        self.image_weight = float(image_weight)
        self.text_weight = float(text_weight)
        _set_or_register_buffer(self, "text_prototypes", torch.empty(0))
        _set_or_register_buffer(self, "image_prototypes", torch.empty(0))
        _set_or_register_buffer(self, "has_image_proto", torch.empty(0, dtype=torch.bool))

    @property
    def embed_dim(self) -> int:
        return _resolve_embed_dim(self.siglip)

    def get_logit_scale(self) -> torch.Tensor:
        return self.siglip.logit_scale.exp().clamp(1.0, 100.0)

    def get_logit_bias(self) -> torch.Tensor:
        return self.siglip.logit_bias.clamp(-10.0, 10.0)

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        _set_or_register_buffer(self, "text_prototypes", F.normalize(prototypes.float(), dim=-1))

    def set_image_prototypes(self, prototypes: torch.Tensor, has_proto: Optional[torch.Tensor] = None) -> None:
        prototypes = F.normalize(prototypes.float(), dim=-1)
        _set_or_register_buffer(self, "image_prototypes", prototypes)
        classes = int(prototypes.size(0))
        if has_proto is None:
            has_proto = torch.ones(classes, dtype=torch.bool, device=prototypes.device)
        else:
            has_proto = has_proto.bool().to(prototypes.device)
        _set_or_register_buffer(self, "has_image_proto", has_proto)

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.siglip.get_image_features(pixel_values=images).float()
        return F.normalize(features, dim=-1)

    def encode_query_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_features(images)

    def compute_image_similarities(self, query_features: torch.Tensor) -> torch.Tensor:
        if self.image_prototypes.ndim == 2:
            return query_features @ F.normalize(self.image_prototypes, dim=-1).t()
        if self.image_prototypes.ndim == 3:
            refs = F.normalize(self.image_prototypes, dim=-1)
            sim_refs = torch.einsum("bd,ckd->bck", query_features, refs)
            if self.image_proto_agg == "attention":
                attn = F.softmax(sim_refs / self.image_proto_agg_tau, dim=-1)
                return (attn * sim_refs).sum(dim=-1)
            return sim_refs.mean(dim=-1)
        raise ValueError(f"Unexpected image prototypes ndim: {self.image_prototypes.ndim}")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("Text prototypes not set.")
        if self.image_prototypes.numel() == 0:
            raise RuntimeError("Image prototypes not set.")
        query = self.encode_image_features(images)
        return self.forward_with_features(query)

    def forward_with_features(self, query_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused/text/image logits from pre-computed query features.

        This avoids re-running the vision encoder when the caller already has
        ``encode_query_features(images)`` available.
        """
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("Text prototypes not set.")
        if self.image_prototypes.numel() == 0:
            raise RuntimeError("Image prototypes not set.")
        prototypes = F.normalize(self.text_prototypes, dim=-1)
        sim_text = query_features @ prototypes.t()
        sim_image = self.compute_image_similarities(query_features)
        scale = self.get_logit_scale()
        logits_text = scale * sim_text
        logits_image = scale * sim_image
        logits_fused = self.image_weight * logits_image + self.text_weight * logits_text
        return logits_fused, logits_text, logits_image


def freeze_all_vision(model: nn.Module) -> None:
    siglip = model.siglip if hasattr(model, "siglip") else model
    for param in siglip.vision_model.parameters():
        param.requires_grad = False


def unfreeze_all_vision(model: nn.Module) -> None:
    siglip = model.siglip if hasattr(model, "siglip") else model
    for param in siglip.vision_model.parameters():
        param.requires_grad = True


def freeze_all_text(model: nn.Module) -> None:
    siglip = model.siglip if hasattr(model, "siglip") else model
    for param in siglip.text_model.parameters():
        param.requires_grad = False


def unfreeze_all_text(model: nn.Module) -> None:
    siglip = model.siglip if hasattr(model, "siglip") else model
    for param in siglip.text_model.parameters():
        param.requires_grad = True


def unfreeze_vision_last_n_layers(model: nn.Module, n_last: int) -> None:
    if n_last <= 0:
        return
    siglip = model.siglip if hasattr(model, "siglip") else model
    layers = siglip.vision_model.encoder.layers
    n_last = min(int(n_last), len(layers))
    for layer in layers[-n_last:]:
        for param in layer.parameters():
            param.requires_grad = True
    for attr_name in ("post_layernorm", "head"):
        module = getattr(siglip.vision_model, attr_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_text_last_n_layers(model: nn.Module, n_last: int) -> None:
    if n_last <= 0:
        return
    siglip = model.siglip if hasattr(model, "siglip") else model
    layers = siglip.text_model.encoder.layers
    n_last = min(int(n_last), len(layers))
    for layer in layers[-n_last:]:
        for param in layer.parameters():
            param.requires_grad = True
    for attr_name in ("final_layer_norm", "head"):
        module = getattr(siglip.text_model, attr_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def get_num_layers(siglip_model, encoder: str = "vision") -> int:
    if encoder == "vision":
        return len(siglip_model.vision_model.encoder.layers)
    if encoder == "text":
        return len(siglip_model.text_model.encoder.layers)
    raise ValueError(f"Unknown encoder: {encoder}")


def get_layer_id_for_siglip(name: str, num_layers: int, encoder: str) -> int:
    token = "vision_model.encoder.layers." if encoder == "vision" else "text_model.encoder.layers."
    if token in name:
        suffix = name.split(token, 1)[1]
        layer_idx = suffix.split(".", 1)[0]
        if layer_idx.isdigit():
            return int(layer_idx) + 1
    if encoder == "vision" and "vision_model.embeddings" in name:
        return 0
    if encoder == "text" and "text_model.embeddings" in name:
        return 0
    return num_layers


def get_trainable_params_info(model: nn.Module) -> Dict[str, float]:
    info: Dict[str, float] = {
        "total_trainable": 0,
        "total_params": 0,
        "vision": 0,
        "text": 0,
        "logit": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        info["total_params"] += int(param.numel())
        if not param.requires_grad:
            continue
        count = int(param.numel())
        info["total_trainable"] += count
        if "vision_model" in name:
            info["vision"] += count
        elif "text_model" in name:
            info["text"] += count
        elif "logit_scale" in name or "logit_bias" in name:
            info["logit"] += count
        else:
            info["other"] += count

    total = float(info["total_params"]) if info["total_params"] else 1.0
    info["trainable_ratio"] = float(info["total_trainable"]) / total
    return info
