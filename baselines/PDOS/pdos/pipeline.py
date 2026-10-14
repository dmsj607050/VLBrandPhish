"""End-to-end PDOS inference and JSON report construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision import make_decision
from .domain_verification import LegalDomainDB, verify_domain
from .feature_extractor import ResNet101FeatureExtractor
from .legal_logo_db import LegalLogoDB
from .preprocessing import crop_box, load_rgb_image
from .similarity import max_reference_similarity


def run_pdos(
    image_path: str | Path,
    url: str,
    detector,
    feature_extractor: ResNet101FeatureExtractor,
    logo_db: LegalLogoDB,
    domain_db: LegalDomainDB,
    similarity_threshold: float = 0.60,
    brand_search: str = "category",
) -> dict[str, Any]:
    detections = detector.predict(image_path)
    image = load_rgb_image(image_path)
    best: dict[str, Any] | None = None
    cache: dict[str, list] = {}
    for detection in detections:
        category = detection["category"]
        if brand_search not in {"category", "global"}:
            raise ValueError("brand_search must be 'category' or 'global'")
        brands = [_category_to_brand(category)] if brand_search == "category" else logo_db.brands()
        query_feature = feature_extractor.extract(crop_box(image, detection["bbox"], padding=0.05))
        for brand in brands:
            reference_paths = logo_db.paths_for(brand)
            if not reference_paths:
                continue
            if brand not in cache:
                cache[brand] = [feature_extractor.extract_path(path) for path in reference_paths]
            score, ref_index = max_reference_similarity(query_feature, cache[brand])
            candidate = {
                **detection,
                "brand": brand,
                "similarity": score,
                "similarity_threshold": similarity_threshold,
                "logo_is_real": score >= similarity_threshold,
                "reference_logo": str(reference_paths[ref_index]),
                "brand_search": brand_search,
            }
            if best is None or candidate["similarity"] > best["similarity"]:
                best = candidate

    predicted_brand = best["brand"] if best else None
    logo_is_real = bool(best and best["logo_is_real"])
    domain = verify_domain(url, predicted_brand or "", domain_db)
    decision = make_decision(predicted_brand, logo_is_real, domain)
    return {
        "input": {"image": str(Path(image_path).resolve()), "url": url},
        "detector": {
            "weights_loaded": bool(getattr(detector, "weights_loaded", False)),
            "weights_source": getattr(detector, "weights_source", None),
            "detections": detections,
        },
        "predicted_brand": predicted_brand,
        "bbox": best["bbox"] if best else None,
        "detector_confidence": best["confidence"] if best else None,
        "cosine_similarity": best["similarity"] if best else None,
        "similarity_threshold": similarity_threshold,
        "logo_is_real": logo_is_real,
        "registered_domain": domain.registered_domain,
        "expected_legal_domains": list(domain.expected_domains),
        "domain_match": domain.matched,
        "domain_hostname": domain.hostname,
        "final_label": decision.label,
        "explanation": decision.reason,
        "homograph_skeleton": domain.homograph_skeleton,
    }


def _category_to_brand(category: str) -> str:
    return category.removesuffix("_1").removesuffix("_2")
