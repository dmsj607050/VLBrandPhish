# Architecture review and design decisions

This document records the source-layout review used for the public VLBrandPhish release.
It compares structural patterns in the local PhishIntention and PhishPedia reference
implementations; it is an engineering review, not a comparison of research results.

## Reader-first layout

The repository is organized around the paths a reader follows when reproducing a result:
method-specific inference under `inference/<method>/`, VLBrandPhish training under
`training/vlbrandphish/`, and paper-oriented analyses under `experiments/`. The root
command is `python cli.py`; the source tree is intentionally flat rather than an
installable package.

Reusable VLBrandPhish model code lives in
`inference/vlbrandphish/components/brand/`. It is intentionally narrow: training and
inference may import it, while experiments must not become a dependency of production
code.

## Decisions

| Risk in a research codebase | VLBrandPhish decision |
| --- | --- |
| Experiments that duplicate method logic | Implemented experiment tools consume saved inference artifacts or public interfaces. Study scaffolds do not claim to be completed evaluators. |
| A wrapper that mixes model loading, cache writes, CLI parsing, and figure rendering | Runtime modules own their inputs; the root CLI dispatches to explicit workflow commands. |
| A wide positional tuple used as implicit configuration | Paths use a named `DefaultPaths` dataclass; record and result structures are dataclasses. |
| Experiment-specific assumptions in production code | Prompt ablations, protocol comparisons, diagnostics, figures, and data preparation live under `experiments/`. |
| Cache files written into source or brand-library folders | Cache locations are explicit command inputs and default under ignored output/cache locations. |
| Duplicate VLBrandPhish model components | `inference/vlbrandphish/components/brand/` owns the shared encoder, fusion, repository, and naming pieces. |
| A hidden dependency on a sibling research repository | The public runtime resolves local data/model paths and does not import sibling repositories at runtime. |
| A study script reporting success with missing methods | Incomplete study runners raise `ExperimentImplementationRequired` before creating output folders. |

## Dependency boundaries

```text
training/vlbrandphish/logo_detection  --> inference/vlbrandphish/components/brand
training/vlbrandphish/brand_recognition/stage1  --\
training/vlbrandphish/brand_recognition/stage2  ---+--> inference/vlbrandphish/components/brand

inference/vlbrandphish/input_loading --> logo_detection --> brand_recognition
                                                          |
                                                          v
                                           domain_verification --> evaluation
                                                          |
                                                          v
                                                       pipeline

experiments/* --> public training/inference interfaces or saved output artifacts
```

The arrows express allowed dependency direction. In particular, production inference and
training code must not import experiments, figures, or annotation tools.

## Contribution rule

Before adding an experiment, decide whether it changes reusable runtime behavior. If it
does, add the component to the narrowest training/inference module and cover it with a
dependency-free contract test where practical. If it only analyses a completed run, keep
it under `experiments/` and document its required input artifacts in
`docs/EXPERIMENTS.md`.

## Release constraint

The repository intentionally excludes screenshots, URLs, trained weights, cached
embeddings, and generated paper outputs. Provide those through a versioned release archive
or a controlled-access channel with hashes and a data-use statement. See
[DATA_AND_MODELS.md](DATA_AND_MODELS.md).