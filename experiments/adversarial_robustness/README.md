# Experiment 4: Adversarial robustness

The runner evaluates PhishEval screenshots with the same detector and fixed-fusion
decision rule used by Experiments 1 to 3. Squatted URLs are labelled phishing; original
URLs are labelled benign. Grouped reports provide detection recall and benign alert rate
for every attack and URL type.

The default source is the locally prepared six perturbation attacks:

    python experiments/adversarial_robustness/run.py

Use the 13 visible-manipulation families with:

    python experiments/adversarial_robustness/run.py --source visible

Use both PhishEval families with:

    python experiments/adversarial_robustness/run.py --source all

The prepared perturbation files are read from the local PhishEval preparation directory.
The runner maps PhishEval aliases to the official 152-brand reference library and, by
default, excludes samples with no available reference brand. The frozen manifest and
experiment_summary.json report this coverage boundary explicitly. Add `--keep-unmapped`
only for a detection-only analysis where top-1 brand accuracy is not meaningful.

Each run writes its manifest, detector output, fixed-fusion predictions, grouped
metrics, logs, and asset hashes to `results/`.
