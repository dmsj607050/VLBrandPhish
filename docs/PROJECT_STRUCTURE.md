# Project structure

The source is organized around the two reader workflows: training a model and running inference. Reusable model components are deliberately outside both workflows so inference never depends on training code.

| Area | Responsibility | Public entry point |
| --- | --- | --- |
| `components/brand/` | Shared SigLIP encoder, fusion layers, brand repository, and brand-name normalization | imported by training and inference |
| `training/logo_detection/` | MMDetection/DINO configuration, dataset construction, loss patching, training, testing, and tiled inference | `vlbrandphish train-detector` |
| `training/brand_recognition/` | Shared brand-training data/model/loss/optimizer code and orchestration | `vlbrandphish train-brand` |
| `training/brand_recognition/common/` | Data, model, loss, optimizer, prompts, transforms, and utilities shared by both stages | imported by Stage-1 and Stage-2 |
| `training/brand_recognition/stage1/` | Pairwise SigLIP training and validation routines | selected by `stage: stage1` |
| `training/brand_recognition/stage2/` | Dual-fusion and TriSig training and validation routines | selected by `stage: stage2` |
| `inference/` | Input loading, detector execution, model/repository loading, brand evidence, domain verification, evaluation, caching, and the end-to-end pipeline | `vlbrandphish detect` / `vlbrandphish infer` |
| `experiments/` | Ablations, analyses, paper figures, and data-preparation studies | module-specific commands |
| `tools/annotation/` | VOC/XML annotation maintenance utilities | module-specific commands |

The dependency direction is intentional: `training/` and `inference/` may import `components/`, but neither may import the other. Experiments may call public workflow modules, but production inference and training modules must not import experiments or annotation tooling.

`scripts/train_brand.py` remains a thin compatibility wrapper, not a second implementation. The detector wrappers configure MMDetection/DINO but do not vendor either upstream project.

All paper-specific outputs, model weights, screenshots, and caches stay outside Git. This keeps source history compact and lets a tagged source revision map cleanly to a separately archived artifact bundle.
