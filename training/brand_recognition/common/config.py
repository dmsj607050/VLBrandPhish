from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``; override values win."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict-like YAML.")

    # Support single-level config inheritance via the ``inherits`` field.
    inherits = cfg.pop("inherits", None)
    if inherits:
        base_ref = Path(inherits)
        base_path = base_ref if base_ref.is_absolute() else (cfg_path.parent / base_ref)
        base = load_config(str(base_path))
        cfg = _deep_merge(base, cfg)

    return cfg


def resolve_stage_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    stage = str(cfg.get("stage", "pipeline")).lower()

    def _find_datasets_root() -> Path:
        candidates = [
            project_root / "data",
            project_root / "datasets",
            project_root.parent / "data",
            project_root.parent / "datasets",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        return candidates[1].resolve()

    datasets_root = _find_datasets_root()

    def _auto_data_root(dataset_type: str) -> Path:
        if dataset_type == "logo2k":
            return datasets_root / "Stage1" / "Logo-2K+"
        if dataset_type == "flat":
            return datasets_root / "Stage2" / "277_brand_pages"
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def _auto_out(stage_name: str) -> Path:
        return project_root / f"outputs_{stage_name}"

    if stage == "pipeline":
        for stage_name in ("stage1", "stage2"):
            stage_cfg = cfg.get(stage_name, {})
            if not isinstance(stage_cfg, dict):
                raise ValueError(f"{stage_name} must be a dict in pipeline config.")
            dataset_type = stage_cfg.get("dataset_type")
            if not dataset_type:
                raise ValueError(f"{stage_name}.dataset_type missing")
            if not stage_cfg.get("data_root"):
                stage_cfg["data_root"] = str(_auto_data_root(str(dataset_type)))
            if not stage_cfg.get("output_dir"):
                stage_cfg["output_dir"] = str(_auto_out(stage_name))
            cfg[stage_name] = stage_cfg
        return cfg

    dataset_type = cfg.get("dataset_type")
    if not dataset_type:
        raise ValueError("dataset_type missing")
    if not cfg.get("data_root"):
        cfg["data_root"] = str(_auto_data_root(str(dataset_type)))
    if not cfg.get("output_dir"):
        cfg["output_dir"] = str(_auto_out(stage))
    return cfg


def split_pipeline_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    common = {k: v for k, v in cfg.items() if k not in {"stage1", "stage2"}}
    cfg_stage1 = dict(common)
    cfg_stage2 = dict(common)
    cfg_stage1.update(cfg["stage1"])
    cfg_stage2.update(cfg["stage2"])
    cfg_stage1["stage"] = "stage1"
    cfg_stage2["stage"] = "stage2"
    return cfg_stage1, cfg_stage2
