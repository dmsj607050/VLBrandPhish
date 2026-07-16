from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

SIMPLE13_TEMPLATES = [
    "{brand}",
    "{brand} logo",
    "logo of {brand}",
    "official logo of {brand}",
    "{brand} brand logo",
    "a photo of the {brand} logo",
    "a clean logo of {brand}",
    "a vector logo of {brand}",
    "{brand} logomark",
    "{brand} wordmark",
    "the {brand} logo on a website",
    "the {brand} logo in a webpage screenshot",
    "a cropped {brand} logo",
]

FIVE_DIM_TEMPLATE_GROUPS = {
    "core_identity": [
        "{brand}",
        "{brand} logo",
        "official logo of {brand}",
        "{brand} brand identity",
    ],
    "mark_type": [
        "{brand} wordmark logo",
        "{brand} logomark",
        "{brand} symbol logo",
        "{brand} emblem",
    ],
    "rendering_style": [
        "clean {brand} logo",
        "flat {brand} vector logo",
        "monochrome {brand} logo",
        "high contrast {brand} brand mark",
    ],
    "scene_context": [
        "{brand} logo on a website header",
        "{brand} logo in a webpage screenshot",
        "{brand} logo on a login page",
        "{brand} logo in navigation bar",
    ],
    "scale_variation": [
        "cropped {brand} logo",
        "small {brand} logo",
        "partial {brand} wordmark",
        "{brand} favicon icon",
    ],
}


def _normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def _normalize_brand_key(text: str) -> str:
    normalized = _normalize_text(text).lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized.replace(" ", "")


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _normalize_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        normalized = _normalize_text(item)
        if normalized:
            out.append(normalized)
    return out


def _resolve_path(path_str: Optional[str], project_root: Optional[Path]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    if project_root is not None:
        return (project_root / path).resolve()
    return path.resolve()


def _brand_in_prompt(prompt: str, brand: str) -> bool:
    return _normalize_brand_key(brand) in _normalize_brand_key(prompt)


def _ensure_brand_name(prompt: str, brand: str) -> str:
    prompt = _normalize_text(prompt)
    if not prompt:
        return ""
    if _brand_in_prompt(prompt, brand):
        return prompt
    return _normalize_text(f"{brand} logo, {prompt}")


def _build_from_templates(brand: str, templates: Iterable[str]) -> List[str]:
    brand_clean = _normalize_text(brand.replace("_", " "))
    prompts = []
    for template in templates:
        normalized = _normalize_text(template)
        if not normalized:
            continue
        prompts.append(normalized.format(brand=brand_clean, b=brand_clean))
    return _dedupe_keep_order(prompts)


def _extract_template_groups(obj: Any) -> Dict[str, List[str]]:
    if isinstance(obj, dict):
        if "template_groups" in obj and isinstance(obj["template_groups"], dict):
            return {
                str(group_name): [str(x) for x in (group_templates or [])]
                for group_name, group_templates in obj["template_groups"].items()
            }
        if "templates" in obj and isinstance(obj["templates"], list):
            return {"default": [str(x) for x in obj["templates"]]}
        extracted: Dict[str, List[str]] = {}
        for key, value in obj.items():
            if isinstance(value, list):
                extracted[str(key)] = [str(x) for x in value]
        if extracted:
            return extracted

    if isinstance(obj, list):
        return {"default": [str(x) for x in obj]}
    raise ValueError("Template config must be a list or dict-like object.")


def _load_template_groups_from_file(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML prompt template files.")
            obj = yaml.safe_load(f)
        elif suffix == ".json":
            obj = json.load(f)
        else:
            raise ValueError(f"Unsupported template file type: {path.suffix}")
    return _extract_template_groups(obj)


def _flatten_template_groups(
    template_groups: Dict[str, List[str]],
    enabled_groups: Optional[List[str]] = None,
) -> List[str]:
    group_names = [str(x) for x in enabled_groups] if enabled_groups else list(template_groups.keys())
    templates: List[str] = []
    for group_name in group_names:
        group_templates = template_groups.get(group_name)
        if group_templates is None:
            raise KeyError(f"Unknown prompt template group: {group_name}")
        templates.extend(group_templates)
    return templates


def _index_generated_prompt_files(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Generated prompts directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Generated prompts path is not a directory: {root}")
    return {_normalize_brand_key(path.stem): path for path in sorted(root.glob("*.jsonl"))}


def _load_generated_prompts_for_brand(
    *,
    brand: str,
    prompt_files: Dict[str, Path],
    prompt_field: str = "prompt",
    top_k: int = 0,
) -> List[str]:
    path = prompt_files.get(_normalize_brand_key(brand))
    if path is None:
        return []

    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            prompt = obj.get(prompt_field)
            if prompt is None:
                continue
            prompts.append(str(prompt))
            if top_k > 0 and len(prompts) >= top_k:
                break
    return _dedupe_keep_order(prompts)


def _index_brand_prompt_text_dirs(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Brand prompt text directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Brand prompt text path is not a directory: {root}")
    return {_normalize_brand_key(path.name): path for path in sorted(root.iterdir()) if path.is_dir()}


def _load_brand_text_prompts_for_brand(
    *,
    brand: str,
    brand_prompt_dirs: Dict[str, Path],
    txt_glob: str = "*.txt",
    expected_line_count: int = 0,
    strict_line_count: bool = False,
    max_lines: int = 0,
) -> List[str]:
    brand_dir = brand_prompt_dirs.get(_normalize_brand_key(brand))
    if brand_dir is None:
        return []

    prompts: List[str] = []
    for txt_file in sorted(brand_dir.glob(txt_glob)):
        with txt_file.open("r", encoding="utf-8-sig") as f:
            for line in f:
                normalized = _normalize_text(line)
                if normalized:
                    prompts.append(normalized)

    if expected_line_count > 0 and strict_line_count and len(prompts) != expected_line_count:
        raise RuntimeError(
            f"Brand '{brand}' expected {expected_line_count} prompts from txt, got {len(prompts)} in {brand_dir}"
        )

    if max_lines > 0:
        prompts = prompts[:max_lines]
    return prompts


def build_label_to_prompts(
    *,
    label_to_name: Dict[int, str],
    cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
    default_prompts: Optional[Dict[int, List[str]]] = None,
) -> Dict[int, List[str]]:
    prompt_cfg = cfg.get("prompts") or {}
    if prompt_cfg and not isinstance(prompt_cfg, dict):
        raise ValueError("cfg['prompts'] must be a dict when provided.")

    if not prompt_cfg:
        if default_prompts is None:
            return {int(k): _build_from_templates(v, SIMPLE13_TEMPLATES) for k, v in label_to_name.items()}
        return {int(k): _dedupe_keep_order(v) for k, v in default_prompts.items()}

    mode = str(prompt_cfg.get("mode", "")).strip().lower()
    preset = str(prompt_cfg.get("preset", "")).strip().lower()
    enabled_groups = prompt_cfg.get("enabled_groups")
    if enabled_groups is not None and not isinstance(enabled_groups, list):
        raise ValueError("prompts.enabled_groups must be a list when provided.")

    force_include_brand_name = bool(prompt_cfg.get("force_include_brand_name", True))
    fallback_to_simple13 = bool(prompt_cfg.get("fallback_to_simple13", True))
    max_prompts_per_brand = int(prompt_cfg.get("max_prompts_per_brand", 0))
    dedupe_prompts = bool(prompt_cfg.get("dedupe_prompts", True))
    include_simple13 = bool(prompt_cfg.get("include_simple13", False))
    include_five_dim = bool(prompt_cfg.get("include_five_dim", False))

    if mode == "simple13":
        include_simple13 = True
    elif mode in {"five_dim", "five_dim_v1"}:
        include_five_dim = True
    elif mode in {"hybrid", "generated_jsonl", "template_file", "brand_txt_dir", "manual_brand_txt", ""}:
        pass
    else:
        raise ValueError(f"Unknown prompts.mode: {mode}")

    if preset == "five_dim_v1":
        include_five_dim = True

    template_file = _resolve_path(prompt_cfg.get("template_file"), project_root)
    generated_dir = _resolve_path(prompt_cfg.get("generated_jsonl_dir"), project_root)
    brand_txt_dir = _resolve_path(prompt_cfg.get("brand_prompt_txt_dir"), project_root)
    generated_top_k = int(prompt_cfg.get("generated_jsonl_top_k", 0))
    prompt_field = str(prompt_cfg.get("generated_jsonl_prompt_field", "prompt"))
    brand_txt_glob = str(prompt_cfg.get("brand_prompt_txt_glob", "*.txt"))
    brand_txt_expected_lines = int(prompt_cfg.get("brand_prompt_txt_expected_lines", 0))
    brand_txt_strict_count = bool(prompt_cfg.get("brand_prompt_txt_strict_count", False))
    brand_txt_top_k = int(prompt_cfg.get("brand_prompt_txt_top_k", 0))

    inline_templates = prompt_cfg.get("templates")
    inline_template_groups = prompt_cfg.get("template_groups")
    file_template_groups = _load_template_groups_from_file(template_file) if template_file is not None else None
    prompt_files = _index_generated_prompt_files(generated_dir) if generated_dir is not None else None
    brand_prompt_dirs = _index_brand_prompt_text_dirs(brand_txt_dir) if brand_txt_dir is not None else None

    resolved: Dict[int, List[str]] = {}
    for label, brand_name in sorted(label_to_name.items()):
        brand_clean = _normalize_text(brand_name.replace("_", " "))
        prompts: List[str] = []

        if include_simple13:
            prompts.extend(_build_from_templates(brand_clean, SIMPLE13_TEMPLATES))
        if include_five_dim:
            prompts.extend(_build_from_templates(
                brand_clean,
                _flatten_template_groups(FIVE_DIM_TEMPLATE_GROUPS, enabled_groups=enabled_groups),
            ))
        if file_template_groups is not None:
            prompts.extend(_build_from_templates(
                brand_clean,
                _flatten_template_groups(file_template_groups, enabled_groups=enabled_groups),
            ))
        if inline_template_groups is not None:
            prompts.extend(_build_from_templates(
                brand_clean,
                _flatten_template_groups(
                    _extract_template_groups({"template_groups": inline_template_groups}),
                    enabled_groups=enabled_groups,
                ),
            ))
        if inline_templates is not None:
            prompts.extend(_build_from_templates(brand_clean, [str(x) for x in inline_templates]))
        if prompt_files is not None:
            prompts.extend(_load_generated_prompts_for_brand(
                brand=brand_clean,
                prompt_files=prompt_files,
                prompt_field=prompt_field,
                top_k=generated_top_k,
            ))
        if brand_prompt_dirs is not None:
            prompts.extend(_load_brand_text_prompts_for_brand(
                brand=brand_clean,
                brand_prompt_dirs=brand_prompt_dirs,
                txt_glob=brand_txt_glob,
                expected_line_count=brand_txt_expected_lines,
                strict_line_count=brand_txt_strict_count,
                max_lines=brand_txt_top_k,
            ))

        if force_include_brand_name:
            prompts = [_ensure_brand_name(prompt, brand_clean) for prompt in prompts]
        prompts = _dedupe_keep_order(prompts) if dedupe_prompts else _normalize_keep_order(prompts)

        if not prompts and default_prompts is not None:
            prompts = _dedupe_keep_order(default_prompts.get(int(label), []))
        if not prompts and fallback_to_simple13:
            prompts = _build_from_templates(brand_clean, SIMPLE13_TEMPLATES)
        if max_prompts_per_brand > 0:
            prompts = prompts[:max_prompts_per_brand]
        if not prompts:
            raise RuntimeError(f"No prompts resolved for label={label}, brand='{brand_clean}'")
        resolved[int(label)] = prompts
    return resolved


def summarize_label_prompts(label_to_prompts: Dict[int, List[str]]) -> Dict[str, float]:
    counts = [len(v) for v in label_to_prompts.values()]
    if not counts:
        return {"classes": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "classes": float(len(counts)),
        "min": float(min(counts)),
        "mean": float(sum(counts) / len(counts)),
        "max": float(max(counts)),
    }
