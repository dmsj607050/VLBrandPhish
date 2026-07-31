# Architecture review and design decisions

This document records the source-layout review used for the public VLBrandPhish release. It compares structural patterns in the local PhishIntention and PhishPedia reference implementations; it is an engineering review, not a comparison of research results.

## What VLBrandPhish adopts

PhishIntention and PhishPedia each keep their core research path easy to locate: logo detection, brand/reference matching, and the final wrapper are visibly separated. VLBrandPhish follows that reader journey directly: each method's inference pipeline lives under `inference/<method>/`, training lives under `training/vlbrandphish/`, and experiments are thin runners under `experiments/` that call shared inference code and load shared model weights.

The repository is organized around the paper's experiments rather than around a single Python package. Code is grouped by method (high cohesion within `inference/<method>/`) and experiments only adapt and orchestrate (low coupling: an experiment directory never re-implements method logic).

## Problems intentionally avoided

| Risk in a research codebase | VLBrandPhish decision |
| --- | --- |
| Experiments that duplicate or fork method logic | Experiments under `experiments/` only call `inference/<method>/` and load `models/<method>/`; they contain only experiment-specific adaptation code. |
| A wrapper that loads models, writes caches, parses CLI arguments, and renders results in one global configuration flow | Runtime components own their own inputs; the public CLI dispatches to explicit commands. |
| A wide positional tuple used as an implicit system configuration | Paths are a named `DefaultPaths` dataclass; record and result structures are dataclasses. |
| Experiment-specific assumptions inside the production path | Prompt ablations, protocol comparisons, diagnostics, paper figures, and data preparation live in `experiments/`. |
| Cache files written into source or brand-library directories | Cache locations are explicit command inputs and default under ignored output/cache locations. |
| Duplicate model components in training and inference | `components/brand/fusion.py` owns the shared residual adapter and logo-type router. |
| Hard dependency on a sibling research repository | Domain policy contains a local, documented implementation and does not load another repository at runtime. |
| Training tools outside the package with separate import conventions | Logo-detector and brand-recognizer training now reside under `training/logo_detection/` and `training/brand_recognition/`. |

## Package boundaries

```text
training/logo_detection  --> components/brand
training/brand_recognition/stage1  --\
training/brand_recognition/stage2  ---+--> components/brand

inference/data_loader --> inference/logo_detection --> inference/brand_recognition
                                                        |
                                                        v
                                     inference/domain_verification --> inference/evaluation
                                                        |
                                                        v
                                               inference/pipeline
```

`components/brand/` is the only shared model layer. The arrows express allowed dependency direction: training and inference may depend on components; inference must not import training; runtime modules must not import experiments or annotation tooling.
## Naming rules

- Production modules use a responsibility name: `runner`, `repository`, `domains`, `policy`, or `encoders`.
- Terms such as `stage1`, `stage2`, dataset protocol names, and paper table labels are retained only where they describe external artifacts or experimental outputs.
- A compatibility script may delegate to a package command, but may not contain an independent implementation.
- New code should expose named data structures rather than positional return tuples.

## Release constraints

The repository intentionally excludes screenshots, URLs, trained weights, cached embeddings, and generated paper outputs. Provide those through a versioned release archive or a controlled-access channel, with hashes and a data-use statement. See [DATA_AND_MODELS.md](DATA_AND_MODELS.md).

## Follow-up rule for contributions

Before adding a new experiment, first decide whether it changes the reusable runtime. If it does not, place it under `experiments/`; if it does, add the reusable component to the narrowest existing runtime package and cover its contract with a dependency-free test where practical.
