@echo off
setlocal EnableExtensions

REM One-click VLBrandPhish runner. The local configuration is deliberately ignored
REM by Git so absolute data paths and model locations never enter the public artifact.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "LOCAL_CONFIG=%SCRIPT_DIR%run_e2e.local.bat"

if not exist "%LOCAL_CONFIG%" (
    echo [VLBrandPhish] Missing local configuration: "%LOCAL_CONFIG%"
    echo Copy run_e2e.local.example.bat to run_e2e.local.bat, then set your paths.
    if not defined VLBRANDPHISH_NO_PAUSE pause
    exit /b 2
)

call "%LOCAL_CONFIG%"
if errorlevel 1 exit /b %errorlevel%

for %%V in (DETECTION_ENV INFERENCE_ENV DET_CONFIG DET_CHECKPOINT BRAND_CHECKPOINT MODEL_DIR LOGO_LIBRARY_DIR DBI5_DIR NORMAL_URLS_CSV BENIGN_ROOT PHISH_ROOT) do (
    if not defined %%V (
        echo [VLBrandPhish] Required variable %%V is not defined in run_e2e.local.bat.
        if not defined VLBRANDPHISH_NO_PAUSE pause
        exit /b 2
    )
)

if not defined RUN_MODE set "RUN_MODE=smoke"
if not defined RUN_NAME set "RUN_NAME=one_click_%RUN_MODE%"
if not defined DEVICE set "DEVICE=cuda:0"
if not defined BRAND_TEXT_MODE set "BRAND_TEXT_MODE=train_templates_mean"
if not defined TOPK set "TOPK=3"
if not defined IMAGE_BATCH_SIZE set "IMAGE_BATCH_SIZE=64"
if not defined TEXT_BATCH_SIZE set "TEXT_BATCH_SIZE=256"

if /I "%RUN_MODE%"=="smoke" (
    set "MAX_BENIGN=1"
    set "MAX_PHISH=1"
    set "MAX_SAMPLES=2"
    set "MAX_LOGO_PER_BRAND=1"
) else if /I "%RUN_MODE%"=="full" (
    set "MAX_BENIGN=-1"
    set "MAX_PHISH=-1"
    set "MAX_SAMPLES=-1"
    set "MAX_LOGO_PER_BRAND=-1"
) else (
    echo [VLBrandPhish] RUN_MODE must be smoke or full, not "%RUN_MODE%".
    if not defined VLBRANDPHISH_NO_PAUSE pause
    exit /b 2
)

set "RUN_DIR=%REPO_ROOT%\outputs\%RUN_NAME%"
set "DETECTION_CSV=%RUN_DIR%\stage1\logo_detection_top3.csv"
set "INFERENCE_DIR=%RUN_DIR%\stage2"
set "CACHE_DIR=%INFERENCE_DIR%\cache"

cd /d "%REPO_ROOT%"
echo.
echo [VLBrandPhish] Stage 1/2 run: %RUN_MODE%
echo [VLBrandPhish] Outputs: %RUN_DIR%
echo.

call conda run -n "%DETECTION_ENV%" python cli.py detect ^
  --det-config "%DET_CONFIG%" ^
  --det-checkpoint "%DET_CHECKPOINT%" ^
  --device "%DEVICE%" ^
  --topk %TOPK% ^
  --benign-root "%BENIGN_ROOT%" ^
  --phish-root "%PHISH_ROOT%" ^
  --max-benign %MAX_BENIGN% ^
  --max-phish %MAX_PHISH% ^
  --out-csv "%DETECTION_CSV%"
if errorlevel 1 goto :failed

echo [VLBrandPhish] Stage 2/2: brand recognition and domain validation
call conda run -n "%INFERENCE_ENV%" python cli.py infer ^
  --det-csv "%DETECTION_CSV%" ^
  --brand-ckpt "%BRAND_CHECKPOINT%" ^
  --model-dir "%MODEL_DIR%" ^
  --logo-library-dir "%LOGO_LIBRARY_DIR%" ^
  --dbi5-dir "%DBI5_DIR%" ^
  --normal-urls-csv "%NORMAL_URLS_CSV%" ^
  --benign-root "%BENIGN_ROOT%" ^
  --phish-root "%PHISH_ROOT%" ^
  --brand-text-mode "%BRAND_TEXT_MODE%" ^
  --max-logo-per-brand %MAX_LOGO_PER_BRAND% ^
  --max-samples %MAX_SAMPLES% ^
  --topk %TOPK% ^
  --image-batch-size %IMAGE_BATCH_SIZE% ^
  --text-batch-size %TEXT_BATCH_SIZE% ^
  --device "%DEVICE%" ^
  --out-dir "%INFERENCE_DIR%" ^
  --cache-dir "%CACHE_DIR%" ^
  --no-progress
if errorlevel 1 goto :failed

echo.
echo [VLBrandPhish] Complete. See "%INFERENCE_DIR%\summary.json".
if not defined VLBRANDPHISH_NO_PAUSE pause
exit /b 0

:failed
echo.
echo [VLBrandPhish] Run failed. Review the command output above.
if not defined VLBRANDPHISH_NO_PAUSE pause
exit /b 1
