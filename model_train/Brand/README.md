# Current brand-recognition training

This package contains the training source and configurations for the current VLBrandPhish brand recognizer. The published experiment uses the SigLIP pairwise objective, initialized from the Logo-2K+ Stage-1 checkpoint, with the final four text-transformer blocks and final two vision-transformer blocks unfrozen. Training uses three DBI-4 prompt descriptions per brand and forces the brand name into each prompt. No router or gate is enabled in this configuration.

The main checkpoint used in the latest experiments is brand_recognition_siglip_text4_vision2_full277.pt. Model weights, training images, and outputs are intentionally excluded from Git.

## Required local files

Place the PhishIntention reference images at data/PhishIntention/277_brand_pages, the 277-brand prompt library at data/VLBrandPhish/06_prompt_library_277, the local SigLIP base model at models/vlbrandphish/siglip-base-patch16-224, and the Logo-2K+ Stage-1 initialization checkpoint at models/vlbrandphish/stage1_logo2k_pretrained.pt.

The active YAML paths are relative to this package directory and assume the repository layout above.

## Train the main model

    cd model_train/Brand
    python train.py --config configs/active/stage1_dbi4_text4_vision2.yaml

Outputs are written under model_train/Brand/outputs/. Use a new output directory or run ID for every experiment; do not overwrite archived checkpoints.

## Reference-count training variants

From the repository root, the ablation runner generates one configuration for each training reference cap and invokes this package's trainer:

    python experiments/ablations/ref_library_size/train_variants.py --run-id refcap_text4_vision2_run01

Optional path overrides are available in train_variants.py. The generated variants use the same seed, prompt source, Stage-1 initialization, and 4-text/2-vision unfreezing as the main model; only the per-brand training-image cap changes.
