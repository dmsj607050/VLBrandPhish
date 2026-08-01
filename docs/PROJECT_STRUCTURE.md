# Project structure

VLBrandPhish uses a flat, experiment-oriented layout. Run commands from the repository
root with `python cli.py ...`; the repository is not distributed as an installed Python
package. The paths below describe the files that actually exist in this release.

| Area | Responsibility | Entry point or status |
| --- | --- | --- |
| `cli.py` | Stable root command dispatcher | `python cli.py {detect,infer,train-brand,train-detector}` |
| `inference/vlbrandphish/` | VLBrandPhish detector, visual-language brand recognition, domain verification, caching, metrics, and end-to-end inference | `python cli.py detect` / `python cli.py infer` |
| `inference/vlbrandphish/components/brand/` | Shared encoders, fusion layers, brand repository, and brand-name normalization used by VLBrandPhish training and inference | imported internally |
| `inference/phishintention/` | Local PhishIntention baseline wrapper and its required support code | wrapper; baseline study evaluator pending |
| `inference/phishpedia/` | Local PhishPedia baseline wrapper and its required support code | wrapper; baseline study evaluator pending |
| `training/vlbrandphish/logo_detection/` | MMDetection/DINO configuration, dataset construction, loss patching, training, testing, and tiled inference | `python cli.py train-detector` |
| `training/vlbrandphish/brand_recognition/` | SigLIP brand-recognizer training orchestration | `python cli.py train-brand` |
| `training/vlbrandphish/brand_recognition/common/` | Data, model, loss, optimizer, prompt, transform, and utility code shared by Stage-1 and Stage-2 | imported by the training stages |
| `training/vlbrandphish/brand_recognition/configs/` | Versioned Stage-1/Stage-2 YAML files and prompt templates | passed to `python cli.py train-brand --config` |
| `experiments/analysis/`, `experiments/ablations/prompt_ablation/`, `experiments/figures/`, `experiments/preparation/` | Implemented post-inference analysis, prompt studies, figures, and preparation tools | see `docs/EXPERIMENTS.md` |
| `experiments/overall_*`, `experiments/logo_detection/`, `experiments/brand_identification/`, `experiments/cross_*`, `experiments/adversarial_robustness/` | Paper-study protocol directories | scaffolds; fail explicitly until full evaluators are ported |
| `data/`, `models/` | Dataset and weight documentation plus ignored local artifacts | see `docs/DATA_AND_MODELS.md` |
| `scripts/` | Artifact verification, manifests, and run metadata | standalone utilities |
| `tests/` | Lightweight, data-free contracts | `python -m unittest discover -s tests/unit -v` |

The dependency direction is deliberate: training and inference may share the narrow
`inference/vlbrandphish/components/brand/` library; production code must not import
experiment or annotation tooling. Experiment code may call public inference/training
interfaces, but it must not silently replace them with a second implementation.

All paper outputs, screenshots, weights, data, and caches stay outside Git. A tagged
source revision must be paired with a separately archived data/model bundle, split
manifests, and run metadata before it can substantiate a paper result.