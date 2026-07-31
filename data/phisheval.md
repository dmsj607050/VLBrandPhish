# PhishEval adversarial samples

Public adversarial dataset for evaluating robustness to logo manipulation and
perturbation attacks. Used for RQ4.

- **Source:** PhishEval (phisheval2024) official release.
  - Repo: `comparative_experiments_B/asversarial experiment/PhishingEval-main` (vendored).
  - Exact public download URL: **TODO** — record the official release link and SHA-256
    before submission.
- **Local copy:** `data/phisheval/` (~21 GB adversarial samples; ~2.3 GB repo)
- **Redistribution:** verify upstream license before redistributing. Squatted-domain
  samples were generated locally with `ail-typo-squatting`
  (https://github.com/typosquatter/ail-typo-squatting).

## Statistics

| Group | Manipulation | Phishing (squatted) | Benign |
| --- | --- | ---:| ---:|
| Original (no manipulation) | — | 1,320 | 110 |
| Visible manipulation | 13 types | 88,152 | 7,346 |
| Perturbation-based | 6 types | 6,528 | 544 |

For each benign URL, 12 typosquatted URLs are generated whose domains differ from the
benign brand's domain, so squatted-phishing counts are 12× the benign counts.

## Attack types

**Visible manipulation (13):** Elimination, ColorReplace, Resizing, Rotation, Integration,
Re-position, Flipping, Replacement, Blurring, Scaling, Omission, FontReplace,
CaseConversion.

**Perturbation (6):** SRNet, ViT-based, Swin-based, FGSM, PGD, CW.

## Role in the paper

- **RQ4:** phishing detection rate on squatted-domain samples under each attack type, plus
  an embedding-space analysis on a 113-sample subset (intra-brand distance and separation
  ratio). Benign samples are used to measure the benign alert rate.

## Evaluation convention

- Benign samples → benign alert rate (false-positive side).
- Squatted-domain phishing samples → phishing detection rate (primary robustness metric).
