# Top-1 fusion-weight ablation

This is the webpage experiment in RQ4, separate from the cropped-logo branch
ablation. It reuses rank-ordered query embeddings and a brand repository cache;
no detector, image encoder, or training run is launched. Only the first crop slot
is used, even if that crop is invalid and a lower-ranked crop is valid.

Use the same frozen metadata, brand-domain CSV and domain-policy implementation
as the reference experiment. Metadata is aligned to the cache by unique image
path, and its class labels are checked. Do not substitute a pre-aggregated Top-3
similarity matrix: the original rank-1 evidence cannot be recovered from it.

From the repository root, with the inference dependencies installed:

```bash
python experiments/ablations/weight_ablation/run.py \
  --query-cache /data/run/stage2_query.npz \
  --repository-cache /data/run/brand_repo.pt \
  --metadata-csv /data/run/per_sample.csv \
  --normal-urls-csv /data/PhishIntention/normal_urls.csv \
  --reference-predictions-csv /data/run/per_sample.csv \
  --url-policy-mode phishintention \
  --device cuda:0 \
  --out-dir outputs/top1_weight_ablation
```

The optional `--domain-rules /data/run/domain.py` freezes the domain-parser
implementation if the reference was produced with a different parser version.
The file must be trusted local code. Its hash is recorded in provenance.

For each image weight in `0,0.2,0.4,0.6,0.8,1`, the script scans thresholds
`0.00..1.00` in steps of `0.01`. Maximum F1 is exploratory because each threshold
is selected on the same evaluation set. Ties prefer higher precision and then a
higher threshold. The default 0.4 image / 0.6 text operating point is retained
independently; its fixed threshold is 0.4 and is also reported for every weight.

When a reference prediction CSV is provided, the default operating point must
match its labels, predicted brands and scores (within floating-point tolerance).
Output includes `summary.csv`, `threshold_grid.csv`, `default_predictions.csv`,
`top1_similarities.npz` and `provenance.json` with input hashes. An existing
nonempty output directory is never overwritten. Raw caches and prediction CSVs
are external research data and must not be committed to this repository.

## Paper result (2026-10-14)

The [aggregate scan](paper_results/summary.csv) covers 46,505 webpages
(24,532 phishing, 21,973 benign) and 277 reference brands.
The [provenance](paper_results/provenance.json) records input hashes.
At image weight 0.4 and threshold 0.4, all alerts and canonical brand predictions
match the saved RQ1 run: TP=23,771, TN=21,786, FP=187, FN=761, F1=0.9804496.
The largest score difference is 5.37e-7 from floating-point arithmetic.

The highest scanned F1 is 0.9821132 at image weight 0.2 and threshold 0.26.
Image weight 0.4 reaches Max F1=0.9807530 at threshold 0.37.
These evaluation-set maxima do not replace the fixed main-experiment settings.
The cached GPU sweep took 12.68 seconds, excluding cache construction and model training.
