# Logo-localization training

This package contains the current DINO/Swin logo-localization training and evaluation source. It is separate from the deployed detector checkpoint used by the paper experiments; the checkpoint is not included in this repository.

## Required local files

The expected PhishIntention annotations and images are:

- data/PhishIntention/logo_train/train_imgs/
- data/PhishIntention/logo_train/train_coco.json
- data/PhishIntention/logo_train/val_imgs/
- data/PhishIntention/logo_train/val_coco.json

Place the required Swin initialization weights at models/vlbrandphish/swin_base_patch4_window12_384_22k.pth. The evaluation detector and its MMDetection config are expected at models/vlbrandphish/detector.pth and models/vlbrandphish/patched_config.py.

## Train and evaluate

    cd model_train/Logo
    python train.py

The trainer writes to model_train/Logo/work_dir/. Check val_is_test and val_ratio in config.py before training: with the current defaults, it splits the training annotations instead of using the separate validation directory. Set val_is_test=False if the independent validation annotations are intended.

Install a compatible PyTorch/MMDetection/MMEngine stack using requirements-detector.txt before running detector training or inference.