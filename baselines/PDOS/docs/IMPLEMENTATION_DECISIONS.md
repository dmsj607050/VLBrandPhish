# Implementation decisions — Phase 1

## Fixed software

- Python: record from the actual run in `docs/PHASE1_REPORT.md`.
- YOLOv5: upstream tag `v7.0`, commit `915bbf294bb74c859f0b41f1c23bc395014ea679`.
- Torchvision ResNet weights: `ResNet101_Weights.IMAGENET1K_V2`, the installed torchvision equivalent of the paper's ImageNet-pretrained ResNet-101.

## Detector architecture

YOLOv5 v7.0's native backbone is `Conv → C3 → SPPF`; its last literal Conv is followed by a C3 block. The paper requires SE immediately before SPPF and describes that location as after the last Conv. This reproduction therefore inserts `SEBlock(1024, reduction=16)` immediately before SPPF, yielding `C3 → SEBlock → SPPF`. The choice is explicit because the paper does not provide modified YAML/source code.

The model YAML uses the paper's 16 logo categories. The downloaded `yolov5s.pt` is a COCO checkpoint, so only shape-compatible shared tensors are loaded into this 16-class model. Its detection head is not treated as a trained logo detector. A proper logo checkpoint requires Object-Train data and training in Phase 2.

The first 10 actual `torch.nn.Conv2d` modules in the YOLOv5s backbone (including convolutional submodules inside C3 blocks, excluding the head) are frozen. The exact module names are emitted to `results/frozen_parameters.txt`. BatchNorm parameters are not silently frozen because the paper specifically says convolutional layers.

## Similarity

- Logo input is resized to 224×224 and normalized with ImageNet mean/std.
- ResNet-101 classification head is removed; the output is 2048-D.
- Multiple legal references for a brand use the maximum cosine similarity. This is a reproduction assumption because the paper does not specify max/mean/top-k aggregation.
- The paper threshold `cosine similarity >= 0.60` is used.

## Domain verification

URLs are parsed with `urllib.parse`; hostnames are normalized with IDNA/UTS-46; eTLD+1 is obtained with `tldextract` using its bundled suffix snapshot with network lookups disabled. Subdomains match through the same registered domain. The stored homograph skeleton is diagnostic only and never turns a mismatched domain into a match.

The runtime supplies minimal in-process compatibility stubs for optional YOLOv5 imports (`IPython`, `psutil`, and `seaborn`) because they are not needed by inference and are absent from the current Python 3.10 environment. No package installation was attempted for these imports.

## Data status

No author-provided dataset was found in the workspace. The single Alibaba screenshot is a demo input, not a paper dataset split. No metrics are hard-coded.

## Supplied reproduction data

For the user's local data, the detector is trained on PhishIntention `Logo_train/train_coco.json` (8,109 images) and evaluated without retraining on Default (PhishIntention val, 901), Merged (Default plus Phishpedia `coco_test`, 2,480), and New (VLBrandPhish `01_phish_12K`, 12,801). All are converted to one class, `logo`, because the cited detector comparison is localization-only. See `docs/PHASE2_TRAINING.md` for commands.
