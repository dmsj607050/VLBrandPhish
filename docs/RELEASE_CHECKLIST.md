# Pre-publication release checklist

- [ ] Replace the placeholder paper title and add author names, venue, year, DOI/arXiv URL, and a `CITATION.cff` file.
- [ ] Confirm that the MIT license is acceptable to every copyright holder and any funder/institutional policy.
- [ ] Review every public file for email addresses, tokens, local paths, crawler logs, browser profiles, private URLs, and unpublished results.
- [ ] Document all third-party code, model, dataset, and brand-asset licenses; cite MMDetection/DINO, SigLIP, and other upstream work in the paper and README as appropriate.
- [ ] Publish model weights and any permitted data through a durable release host with version, checksum, and access terms.
- [ ] Add a minimal example or a data-access script that can run without private data.
- [ ] Run `python -m compileall src scripts training` and `python -m unittest discover -s tests -v`.
- [ ] Create a clean Git commit, tag the exact paper-release revision (for example `v1.0.0`), and make a GitHub Release.
