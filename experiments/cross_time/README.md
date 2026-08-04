# Experiment 2: Cross-time evaluation

This runner evaluates the held-out recent phishing samples from brands seen while the
brand encoder was trained. It reads the versioned split in
`08_brands_overlap.csv` and selects rows with `in_277` equal to 1.

Run from the repository root:

    python experiments/cross_time/run.py

The split has exactly 48 seen-training brands and 6,400 phishing samples. All readable
benign screenshots are included. Each input manifest records collection_timestamp from
the source JSON when it is available, so the temporal provenance remains auditable.

Outputs use the same layout as Experiment 1 and use the identical fixed fusion and
threshold; this script does not search weights or thresholds.
