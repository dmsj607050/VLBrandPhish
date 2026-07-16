from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...recognition.fusion import LogoTypeRouter, ResidualAdapter


@dataclass
class ProtoCache:
    prototypes: torch.Tensor
    label_to_prompts: Dict[int, List[str]]
    labels: Optional[List[int]] = None
    label_to_index: Optional[Dict[int, int]] = None


@dataclass
class TriSigInferenceConfig:
    use_text_branch: bool = True
    use_image_branch: bool = True
    use_router: bool = True
    fixed_weights: Optional[Tuple[float, float]] = None
    normalize_active_weights: bool = True
    return_branch_scores: bool = True
    return_route_weights: bool = True
    return_uncertainty: bool = True


def _set_or_register_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
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


class ClassAwareGate(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.proto_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        last = self.gate_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        *,
        query: torch.Tensor,
        text_proto: torch.Tensor,
        image_proto: torch.Tensor,
        has_image_proto: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = query.size(0)
        classes = text_proto.size(0)

        q = self.query_proj(query).unsqueeze(1).expand(batch, classes, -1)
        text_key = self.proto_proj(text_proto).unsqueeze(0).expand(batch, classes, -1)
        image_key = self.proto_proj(image_proto).unsqueeze(0).expand(batch, classes, -1)

        text_feat = q * text_key
        image_feat = q * image_key
        alpha = torch.sigmoid(self.gate_head(torch.cat([image_feat, text_feat], dim=-1)).squeeze(-1))

        if has_image_proto is not None:
            alpha = alpha * has_image_proto.unsqueeze(0).float()
        return alpha


class SigLIPDualFusionModel(nn.Module):
    def __init__(
        self,
        siglip_model,
        *,
        use_adapter: bool = True,
        adapter_hidden_dim: int = 256,
        adapter_dropout: float = 0.1,
        use_class_aware_gate: bool = True,
        gate_hidden_dim: int = 256,
        gate_dropout: float = 0.1,
        image_proto_agg: str = "attention",
        image_proto_agg_tau: float = 0.07,
    ):
        super().__init__()
        self.siglip = siglip_model
        embed_dim = _resolve_embed_dim(self.siglip)
        self.use_class_aware_gate = bool(use_class_aware_gate)
        self.image_proto_agg = str(image_proto_agg).lower()
        self.image_proto_agg_tau = float(image_proto_agg_tau)
        self.adapter = None
        if bool(use_adapter):
            self.adapter = ResidualAdapter(
                embed_dim=embed_dim,
                hidden_dim=int(adapter_hidden_dim),
                dropout=float(adapter_dropout),
            )
        self.gate = ClassAwareGate(
            embed_dim=embed_dim,
            hidden_dim=int(gate_hidden_dim),
            dropout=float(gate_dropout),
        )
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
        query = self.encode_image_features(images)
        if self.adapter is not None:
            query = F.normalize(query + self.adapter(query), dim=-1)
        return query

    def compute_similarities(self, query_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim_text = query_features @ F.normalize(self.text_prototypes, dim=-1).t()
        if self.image_prototypes.ndim == 2:
            sim_image = query_features @ F.normalize(self.image_prototypes, dim=-1).t()
        elif self.image_prototypes.ndim == 3:
            refs = F.normalize(self.image_prototypes, dim=-1)
            sim_refs = torch.einsum("bd,ckd->bck", query_features, refs)
            if self.image_proto_agg == "attention":
                attn = F.softmax(sim_refs / self.image_proto_agg_tau, dim=-1)
                sim_image = (attn * sim_refs).sum(dim=-1)
            else:
                sim_image = sim_refs.mean(dim=-1)
        else:
            raise ValueError(f"Unexpected image prototypes ndim: {self.image_prototypes.ndim}")
        return sim_text, sim_image

    def forward(self, images: torch.Tensor, return_gate_weights: bool = True):
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("Text prototypes not set.")
        if self.image_prototypes.numel() == 0:
            raise RuntimeError("Image prototypes not set.")

        query = self.encode_query_features(images)
        sim_text, sim_image = self.compute_similarities(query)
        scale = self.get_logit_scale()
        logits_text = scale * sim_text
        logits_image = scale * sim_image

        if self.image_prototypes.ndim == 3:
            image_proto_for_gate = F.normalize(self.image_prototypes.mean(dim=1), dim=-1)
        else:
            image_proto_for_gate = F.normalize(self.image_prototypes, dim=-1)
        text_proto_for_gate = F.normalize(self.text_prototypes, dim=-1)

        gate_weights = self.gate(
            query=query,
            text_proto=text_proto_for_gate,
            image_proto=image_proto_for_gate,
            has_image_proto=self.has_image_proto if self.has_image_proto.numel() > 0 else None,
        )
        logits_fused = gate_weights * logits_image + (1.0 - gate_weights) * logits_text

        if return_gate_weights:
            return logits_fused, logits_text, logits_image, gate_weights
        return logits_fused, logits_text, logits_image, None


class SigLIPTriSigModel(nn.Module):
    def __init__(
        self,
        siglip_model,
        *,
        use_adapter: bool = True,
        adapter_hidden_dim: int = 256,
        adapter_dropout: float = 0.1,
        router_hidden_dim: int = 256,
        router_dropout: float = 0.1,
        image_proto_agg: str = "attention",
        image_proto_agg_tau: float = 0.07,
    ):
        super().__init__()
        self.siglip = siglip_model
        embed_dim = _resolve_embed_dim(self.siglip)
        self.image_proto_agg = str(image_proto_agg).lower()
        self.image_proto_agg_tau = float(image_proto_agg_tau)
        self.uncertainty_temperature = 1.0
        self.adapter = None
        if bool(use_adapter):
            self.adapter = ResidualAdapter(
                embed_dim=embed_dim,
                hidden_dim=int(adapter_hidden_dim),
                dropout=float(adapter_dropout),
            )
        self.router = LogoTypeRouter(
            embed_dim=embed_dim,
            hidden_dim=int(router_hidden_dim),
            dropout=float(router_dropout),
            stats_dim=6,
            out_dim=2,
        )
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
        query = self.encode_image_features(images)
        if self.adapter is not None:
            query = F.normalize(query + self.adapter(query), dim=-1)
        return query

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

    def _branch_uncertainty_features(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = F.softmax(logits / float(self.uncertainty_temperature), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        if logits.size(-1) > 1:
            top2 = logits.topk(k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones(logits.size(0), device=logits.device, dtype=logits.dtype)
        max_prob = probs.max(dim=-1).values
        uncertainty = entropy / math.log(max(2, logits.size(-1))) if logits.size(-1) > 1 else torch.zeros_like(entropy)
        confidence = 0.5 * max_prob + 0.5 * torch.sigmoid(margin)
        return {
            "entropy": entropy,
            "margin": margin,
            "max_prob": max_prob,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }

    def estimate_branch_uncertainty(
        self,
        logits_text: torch.Tensor,
        logits_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_stats = self._branch_uncertainty_features(logits_text)
        image_stats = self._branch_uncertainty_features(logits_image)
        agreement = torch.sigmoid((logits_text.softmax(dim=-1) * logits_image.softmax(dim=-1)).sum(dim=-1))
        branch_stats = torch.stack(
            [
                text_stats["uncertainty"],
                image_stats["uncertainty"],
                text_stats["confidence"],
                image_stats["confidence"],
                torch.abs(text_stats["margin"] - image_stats["margin"]),
                agreement,
            ],
            dim=-1,
        )
        return {
            "text": text_stats,
            "image": image_stats,
            "agreement": agreement,
            "router_features": branch_stats,
        }

    def _build_fusion_weights(
        self,
        *,
        query: torch.Tensor,
        branch_stats: torch.Tensor,
        cfg: TriSigInferenceConfig,
    ) -> torch.Tensor:
        active_mask = torch.tensor(
            [cfg.use_text_branch, cfg.use_image_branch],
            dtype=query.dtype,
            device=query.device,
        ).unsqueeze(0).expand(query.size(0), -1)

        if cfg.use_router:
            weights = self.router(query, branch_stats)
        else:
            if cfg.fixed_weights is None:
                base = torch.ones(2, dtype=query.dtype, device=query.device)
            else:
                base = torch.tensor(cfg.fixed_weights, dtype=query.dtype, device=query.device)
            weights = base.unsqueeze(0).expand(query.size(0), -1)

        weights = weights * active_mask
        if cfg.normalize_active_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights

    def forward(
        self,
        images: torch.Tensor,
        *,
        inference_cfg: Optional[TriSigInferenceConfig] = None,
        return_route_weights: bool = True,
    ):
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("Text prototypes not set.")
        if self.image_prototypes.numel() == 0:
            raise RuntimeError("Image prototypes not set.")

        cfg = inference_cfg or TriSigInferenceConfig(return_route_weights=return_route_weights)
        query = self.encode_query_features(images)
        prototypes = F.normalize(self.text_prototypes, dim=-1)
        sim_text = query @ prototypes.t()
        sim_image = self.compute_image_similarities(query)

        scale = self.get_logit_scale()
        logits_text = scale * sim_text
        logits_image = scale * sim_image

        if not cfg.use_text_branch:
            logits_text = torch.zeros_like(logits_text)
        if not cfg.use_image_branch:
            logits_image = torch.zeros_like(logits_image)

        uncertainty = self.estimate_branch_uncertainty(logits_text, logits_image)
        route_weights = self._build_fusion_weights(
            query=query,
            branch_stats=uncertainty["router_features"],
            cfg=cfg,
        )
        logits_fused = (
            route_weights[:, 0:1] * logits_text
            + route_weights[:, 1:2] * logits_image
        )

        should_return_weights = return_route_weights or cfg.return_route_weights
        if should_return_weights:
            if cfg.return_uncertainty:
                return logits_fused, logits_text, logits_image, route_weights, uncertainty
            return logits_fused, logits_text, logits_image, route_weights, None
        if cfg.return_uncertainty:
            return logits_fused, logits_text, logits_image, None, uncertainty
        return logits_fused, logits_text, logits_image, None, None


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
    for module in (siglip.vision_model.post_layernorm, siglip.vision_model.head):
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
    for module in (siglip.text_model.final_layer_norm, siglip.text_model.head):
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
        "gate": 0,
        "router": 0,
        "adapter": 0,
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
        elif "gate" in name:
            info["gate"] += count
        elif "router" in name:
            info["router"] += count
        elif "adapter" in name:
            info["adapter"] += count
        elif "logit_scale" in name or "logit_bias" in name:
            info["logit"] += count
        else:
            info["other"] += count

    total = float(info["total_params"]) if info["total_params"] else 1.0
    info["trainable_ratio"] = float(info["total_trainable"]) / total
    return info
