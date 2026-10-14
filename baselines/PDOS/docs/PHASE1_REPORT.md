# PDOS Phase 1 report

## Status

The core code path, model architecture, local weight loading, and end-to-end CLI smoke test pass locally. The demo executes, but its prediction is not a meaningful Logo result yet because the available `yolov5s.pt` is COCO-pretrained and the 16-class Logo head has not been trained.

## 1. Environment

- OS/runtime: Windows, Python 3.10.10
- PyTorch: 2.5.1+cu121
- torchvision: 0.20.1+cu121
- CUDA runtime: 12.1; CUDA available: true
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Original paper hardware: RTX 3090; this is a different local machine.

## 2. Fixed YOLOv5 code

- Repository: `ultralytics/yolov5`
- Tag: `v7.0`
- Commit: `915bbf294bb74c859f0b41f1c23bc395014ea679`
- Checkout: `third_party/yolov5/`

## 3. Model architecture

The custom 16-class YAML is `configs/pdos_yolov5s_se.yaml`. The verified backbone sequence is:

`... → C3 → SEBlock(512, reduction=16) → SPPF`

The actual layer listing is saved in `results/model_structure.txt`; SE is layer 9 and SPPF is layer 10. YOLOv5s width scaling makes this tensor 512 channels, which is why the implementation uses 512 rather than the unscaled YAML channel value 1024.

## 4. Frozen layers

The first 10 actual `Conv2d` modules in the backbone are frozen. The exact names are saved in `results/frozen_parameters.txt`. The freeze is limited to convolution parameters; BatchNorm parameters are not silently frozen.

## 5. ResNet-101

`pdos/feature_extractor.py` removes the classification head from ImageNet-pretrained ResNet-101 and returns a 2048-D vector. The locally supplied `models/resnet101-cd907fc2.pth` was loaded successfully, and a forward pass on the provided screenshot crop returned shape `(2048,)` on CUDA.

## 6. Similarity and domain verification

- Logo preprocessing: 224×224, ImageNet normalization.
- Cosine similarity: L2-normalized feature dot product.
- Threshold: `0.60`.
- Multiple references: maximum similarity; this is explicitly recorded as a reproduction assumption.
- Domain parsing: URL parsing, IDNA/UTS-46 normalization, registered-domain/eTLD+1 comparison, subdomain support, and diagnostic homograph skeleton.
- `tldextract` is forced to its packaged suffix snapshot; network lookups are disabled.

The local smoke checks passed for `www.alibaba.com → alibaba.com` and for the mismatching domain `alibaba.example.com`.

## 7. Provided test image

The image `Alibaba_test.png` was inspected locally. It is a full-page Alibaba screenshot. Since no URL was supplied with the image, the demo used the explicit example URL `https://www.alibaba.com/`. Its output is saved at `results/predictions/alibaba_demo.json`.

## 8. Blocking issue

The originally attempted YOLOv5 weight download failed:

`curl: (7) Failed to connect to github.com:443 over proxy 127.0.0.1`

The user subsequently supplied `models/yolo/yolov5su.pt`, but it is an Ultralytics-format checkpoint. Loading it requires the absent `ultralytics` package and it is not compatible with the fixed YOLOv5 v7.0 architecture; it was not used.

The correct `models/yolo/yolov5s.pt` is now present and loads successfully. It initializes 186 shape-compatible tensors from the shared COCO model, but the 16-class detection head and SE parameters are not trained. No trained 16-class Logo checkpoint or author dataset is present. The current run therefore returned zero Logo detections and the conservative phishing decision “No verified legal logo and no matching legal domain were found.”

Therefore the acceptance item “meaningful Logo detection on the Alibaba image” remains blocked pending Object-Train data and detector training, or a user-provided 16-class Logo checkpoint. Phase 2 training is blocked until the paper data or a documented reconstructed dataset is supplied.
