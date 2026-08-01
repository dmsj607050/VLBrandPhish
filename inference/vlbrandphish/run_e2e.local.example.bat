@echo off
REM Copy this file to run_e2e.local.bat. It is intentionally ignored by Git.
REM Use RUN_MODE=smoke first; change it to full only after the smoke run succeeds.
set "RUN_MODE=smoke"
set "RUN_NAME=one_click_smoke"
set "DETECTION_ENV=your_mmdetection_conda_environment"
set "INFERENCE_ENV=your_siglip_conda_environment"
set "DEVICE=cuda:0"

set "DET_CONFIG=path\to\detector_config.py"
set "DET_CHECKPOINT=path\to\logo_detector.pth"
set "BRAND_CHECKPOINT=path\to\brand_recognizer.pt"
set "MODEL_DIR=path\to\siglip-base-patch16-224"
set "LOGO_LIBRARY_DIR=path\to\reference_logo_library"
set "DBI5_DIR=path\to\brand_prompt_directory"
set "NORMAL_URLS_CSV=path\to\normal_urls.csv"
set "BENIGN_ROOT=path\to\benign_screenshots"
set "PHISH_ROOT=path\to\phishing_screenshots"