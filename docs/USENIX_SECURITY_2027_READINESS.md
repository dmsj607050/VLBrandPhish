# USENIX Security '27 submission and artifact readiness

This document is a submission-readiness checklist for VLBrandPhish. It reflects the [USENIX Security '27 preliminary CFP](https://www.usenix.org/conference/usenixsecurity27/call-for-papers), accessed on 2026-07-31. The final CFP and HotCRP instructions prevail if they differ.

## Deadlines and submission constraints

For Cycle 1, mandatory registration of the fixed title, complete author list (with ORCIDs), non-blank tentative abstract, and topics is due **2026-08-18 23:59 AoE**. The paper is due **2026-08-25 23:59 AoE** and submission artifacts are due **2026-08-28 23:59 AoE**. The main body is limited to 13 pages, in the USENIX Security two-column template; references and appendices are outside the main-body limit. A final paper is limited to 20 pages total.

The paper must include an **Open Science** appendix. An ethics appendix is strongly encouraged, and is particularly important here because phishing screenshots, URLs, brands, and web collection can involve privacy, malicious content, intellectual-property, and takedown risks.

## Double-blind artifact rule

The public project repository is **not** suitable as the review artifact: its repository name, account, pull requests, Git history, and timestamps may identify the authors. For blind review, create a fresh artifact mirror with:

- no Git history, usernames, organization names, email addresses, ORCIDs, or tracking links;
- no author names in README, package metadata, model cards, archive metadata, output files, screenshots, or PDFs;
- an anonymous, untracked URL available throughout review (the CFP recommends anonymous.4open.science with conference ID `SEC27`);
- a stable archive layout, exact commands, input access instructions, and SHA-256 manifest.

Do not add anonymous-review URLs to this public repository before the review process ends. Put the review URL only in the paper's Open Science appendix.

## Current repository assessment

| Requirement | Current status | Required before submission |
| --- | --- | --- |
| Clean source layout and root CLI entry points | Partial pass | Keep the flat root-CLI layout; use a fresh anonymous mirror. |
| Anonymous artifact | Blocked | Export a new history-free, identity-free mirror and inspect archive metadata. |
| Reproducible dependency specification | Partial pass | Record exact Python, PyTorch, CUDA, MMDetection, Transformers, and GPU versions used for every reported table. |
| Source/configuration availability | Partial pass | Include all code and frozen YAML configurations used for claimed results. |
| Model, data, and benchmark access | Blocked | Publish or provide anonymous controlled access to checkpoints, brand library, evaluation inputs, split files, licenses, and SHA-256 values; explain any lawful restriction. |
| One-command evaluation of each reported table | Blocked | Add paper-table command scripts and expected metric files after final experiments are frozen. |
| Sanity checks and artifact integrity | Pass for repository checks | Run `python scripts/verify_artifact.py` and create a manifest for the exact artifact archive. |
| Open Science appendix | Blocked | Add a concise anonymous access statement to the paper, enumerating source, configs, models, data, and evaluation scripts. |
| Ethics discussion | Blocked | State collection, storage, access control, harmful-content handling, URL redaction, brand/IP treatment, and responsible disclosure/takedown procedures. |
| Novelty, rigor, and evidence | Not assessable from code | Demonstrate a clear threat model, fair baseline reproduction, split hygiene, ablations, robustness/failure analysis, and claims supported by confidence intervals or repeated-run statistics. |

## Artifact preparation procedure

1. Freeze the commit and record its full hash privately. Do not expose that hash in the anonymous artifact if it can be linked to this public repository.
2. Copy the release source into a new directory without `.git/`, GitHub workflow run links, author metadata, or local result folders.
3. Add the exact environment specification and all commands needed for the paper's tables and figures. Record hardware and random seeds.
4. Include permitted data/checkpoints or an anonymous access mechanism. For every unavailable artifact, explain the restriction and supply all remaining reproducible components.
5. Run the artifact checker from the artifact root:

   ```bash
   python scripts/verify_artifact.py
   ```

6. Generate a deterministic checksum manifest, including only files that reviewers are allowed to access:

   ```bash
   python scripts/create_artifact_manifest.py --tracked-only \
     --include README.md --include LICENSE --include pyproject.toml --include cli.py \
     --include data --include models --include docs --include experiments \
     --include inference --include training --include scripts \\
     --output artifact_manifest.json
   ```

7. Inspect the archive manually for identity leaks and run all table-reproduction commands in a clean environment.

## What this repository cannot prove

Passing code checks does not establish USENIX Security acceptance readiness. The paper must independently establish originality, relevance to systems security, scientific correctness, clarity, an appropriate threat model, and strong empirical evidence. In particular, a phishing-detection paper should make all train/validation/test boundaries, deduplication rules, temporal collection decisions, baseline tuning budgets, and error cases auditable.