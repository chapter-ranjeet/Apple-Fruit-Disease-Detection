# Project Progress

## Phase 1 - Project initialization

**Status:** Complete

### Completed

- Inspected the repository; it contained no project files or dataset.
- Added the initial project directory structure.
- Added Python dependency requirements targeting Python 3.11.
- Added configurable paths and model settings in `config/settings.yaml`.
- Added an Ultralytics dataset configuration template in `data.yaml`.
- Added Git ignore rules for environments, datasets, weights, caches, and outputs.
- Added installation, dataset-layout, and workflow documentation in `README.md`.
- Checked the local Python environment: Python 3.14.7 is available.
- Attempted GPU detection; `nvidia-smi` could not query the GPU because of
  insufficient permissions, so CUDA availability remains unverified.

### Files created

- `.gitignore`
- `README.md`
- `PROJECT_PROGRESS.md`
- `requirements.txt`
- `data.yaml`
- `config/settings.yaml`
- `app/.gitkeep`
- `dataset/.gitkeep`
- `models/.gitkeep`
- `notebooks/.gitkeep`
- `results/.gitkeep`
- `src/.gitkeep`

## Phase 2 - Dataset inspection and exploratory analysis

**Status:** Complete with blocking findings

### Dataset inspected

- Source: `C:\Users\ACER\Downloads\datasets\ds`
- Format: folder-based image classification (`Train` and `Test`)
- Images: 5,566 total; 3,961 train and 1,605 test
- Readable images: 5,566
- Corrupted/unreadable images: 0
- Classes discovered from actual image folders:
  - `Blotch_Apple`: 1,514
  - `Normal_Apple`: 973
  - `Rot_Apple`: 1,891
  - `Scab_Apple`: 1,188
- Dimensions: width 89-2,112 pixels; height 100-2,816 pixels
- Detection annotation files: 0
- Exact duplicate groups: 317
- Exact duplicate groups crossing splits: 6
- Potential near-duplicate pairs: 92

### Completed

- Added modular format detection and read-only dataset analysis.
- Added YOLO annotation validation, image quality screening, and duplicate detection scripts.
- Generated statistics, CSV/JSON outputs, plots, and a Markdown report.
- Added unit tests for format detection, YOLO validation, and perceptual hashing.
- Confirmed no model training was started.
- Original dataset files were not modified.

### Generated files

- `src/dataset_analysis.py`
- `src/annotation_validator.py`
- `src/image_quality_analysis.py`
- `src/duplicate_detector.py`
- `tests/test_dataset_analysis.py`
- `results/dataset_analysis/dataset_summary.json`
- `results/dataset_analysis/dataset_summary.csv`
- `results/dataset_analysis/class_distribution.csv`
- `results/dataset_analysis/annotation_errors.csv`
- `results/dataset_analysis/corrupted_images.csv`
- `results/dataset_analysis/duplicate_images.csv`
- `results/dataset_analysis/potential_near_duplicates.csv`
- `results/dataset_analysis/image_quality_report.csv`
- `results/dataset_analysis/plots/`
- `results/dataset_analysis/DATASET_ANALYSIS_REPORT.md`

### Blocking findings and pending issues

- The dataset has no bounding-box annotations, so it is **NOT READY** for the requested YOLO object-detection workflow.
- Six exact duplicate groups cross train/test and require review before evaluation.
- The 92 near-duplicate pairs are screening candidates and require manual review.
- Quality flags are screening indicators, not definitive judgments; review the generated CSV before preprocessing.

### Phase 3 - Multi-view data preparation

**Status:** Complete with warnings

### Completed

- Re-verified the Phase 2 reports and the source dataset.
- Grouped 5,566 images into 5,246 exact SHA-256 groups.
- Identified 290 exact-hash groups with conflicting source class labels; no
  label was silently selected.
- Created the derived `dataset_processed/` classification dataset by copying
  files; originals were not moved, deleted, or edited.
- Created a deterministic group-aware split with seed 42:
  - train: 3,897
  - val: 835
  - test: 834
- Preserved all four classes in every derived split.
- Verified exact duplicate groups never cross derived splits.
- Copied potential near-duplicate candidates into the Phase 3 report for
  separate manual review; they were not treated as definitive same-apple groups.
- Added RGB loading, validation, resize, normalization, and training-only
  augmentation configuration.
- Added a strict future four-view schema and documented fusion design.
- Added Phase 3 readiness and multi-view limitation reports.
- Added and passed Phase 3 tests.
- No model training was started.

## Phase 3.1 - Dataset label conflict audit

**Status:** Complete with warnings

### Audit result

- Conflicting exact-hash groups audited: 290
- Conflicting images audited: 580
- Verified groups: 0
- Unresolved groups: 290
- SHA-256 verification failures: 0
- Resolution: all unresolved groups quarantined; no majority label selected

### Rebuilt derived dataset

- Images retained: 4,986
- Images quarantined: 580
- Train: 3,491
- Validation: 748
- Test: 747
- Exact duplicate leakage across derived splits: 0
- Conflicting groups remaining in derived data: 0
- All four classes remain represented in all splits.

### Generated files

- `src/label_conflict_audit.py`
- `results/phase3_1/label_conflict_audit.csv`
- `results/phase3_1/verified_groups.csv`
- `results/phase3_1/unresolved_groups.csv`
- `results/phase3_1/quarantine_manifest.csv`
- `results/phase3_1/conflict_resolution_report.md`
- `results/phase3_1/audit_summary.json`
- `results/phase3_1/split_summary.json`
- `results/phase3_1/split_manifest.csv`
- `results/phase3_1/duplicate_resolution_report.csv`
- `results/phase3_1/preprocessing_report.json`
- `results/phase3_1/dataset_readiness_report.md`

### Remaining requirements

- Review the 290 quarantined groups if additional labeled data is required.
- Review 92 potential near-duplicate candidates and image-quality flags.
- Collect reviewed apple-level four-view records before multi-view evaluation.
- The source remains classification-only; YOLO detection still requires
  bounding-box annotations.

### Phase 3 generated files

- `src/preprocessing.py`
- `src/duplicate_resolution.py`
- `src/split_dataset.py`
- `src/multi_view_schema.py`
- `src/dataset_readiness.py`
- `tests/test_phase3.py`
- `dataset_processed/train/`, `dataset_processed/val/`, `dataset_processed/test/`
- `results/phase3/split_summary.json`
- `results/phase3/split_manifest.csv`
- `results/phase3/duplicate_resolution_report.csv`
- `results/phase3/potential_near_duplicates.csv`
- `results/phase3/preprocessing_report.json`
- `results/phase3/dataset_readiness_report.md`
- `results/phase3/multi_view_dataset_report.md`

### Dataset limitations and remaining requirements

- The source is image classification data, not object detection data.
- No reliable metadata or evidence establishes four photographs of the same
  physical apple.
- No `apple_id` or reviewed front/back/upper/lower records were fabricated.
- True multi-view evaluation requires reviewed apple-level records with
  disease ground truth and a leakage-safe apple-level split.
- Near-duplicate candidates and quality flags still require manual review.

### Verification

- Source file count remains 5,566.
- All derived images validated successfully.
- Exact duplicate leakage across derived splits: 0.
- Unit tests: 8 passed.
- Python compilation and module imports: passed.
- Classification readiness: not ready for training until conflicting labels
  are reviewed or quarantined.
- Multi-view readiness: not ready.
- YOLO detection readiness: not ready without bounding-box annotations.

## Phase 4 - YOLO11 classification training

**Status:** Blocked before training by readiness gates

### Readiness checks performed

- Phase 2, Phase 3, and Phase 3.1 reports reviewed.
- Current cleaned manifest verified: 4,986 images with the expected four classes.
- Exact duplicate leakage verified at zero.
- 92 perceptual near-duplicate candidates reviewed programmatically.
- 39 candidates cross train/validation/test boundaries.
- 73 affected images identified for quarantine pending human review.
- Prospective post-quarantine population: 4,913 images
  (3,458 train, 732 validation, 723 test).
- Python 3.11 is not installed; only Python 3.14.7 is available.
- PyTorch and Ultralytics are not installed.
- GPU query is permission-limited.

### Decision

Training was **not started**. No model metrics, checkpoints, or claims were
fabricated. The exact near-duplicate quarantine is documented in
`results/phase4/near_duplicate_quarantine_manifest.csv`, and the complete
decision is documented in `results/phase4/TRAINING_READINESS_REPORT.md`.

### Required before training

1. Human review or approval of the 39 cross-split near-duplicate pairs.
2. Python 3.11 installation and compatible virtual environment.
3. Compatible PyTorch and Ultralytics installation.
4. Re-run the complete readiness gate using the final approved manifest.

## Phase 4.1 - Human review of perceptual near-duplicates

**Status:** Pending human decisions

### Completed

- Re-verified all 39 cross-split candidate pairs against the source dataset
  and Phase 3.1 manifest.
- Confirmed 73 unique affected images.
- Generated 39 side-by-side review sheets.
- Generated a blank review template with no automatic decisions.
- Added a strict resolution script that rejects incomplete/invalid review
  decisions and never infers labels from similarity.
- Added tests for candidate discovery and explicit review handling.
- No source files were changed, no dataset was rebuilt, and no training was
  started.

### Generated files

- `src/near_duplicate_review.py`
- `tests/test_phase4_1.py`
- `results/phase4_1/review_sheets/`
- `results/phase4_1/near_duplicate_human_review.csv`
- `results/phase4_1/NEAR_DUPLICATE_REVIEW_REPORT.md`
- `results/phase4_1/final_dataset_summary.json`
- `results/phase4_1/final_manifest.csv`

### Pending

- Human decisions for all 39 pairs.
- Reviewer notes, reviewer identity, and review timestamps.
- Rebuild and validate the final approved dataset after decisions are
  supplied.

## Phase 4.2 - Final dataset reconstruction and validation

**Status:** Complete with warnings

### Approved review audit

- Review rows: 39
- Pair IDs: ND-001 through ND-039, unique and complete
- Reviewer: Ranjeet
- Review date: 2026-11-01
- Approved rows: 39
- Decisions: 9 `same_image`, 28 `different_apple`, 2 `uncertain`,
  0 `same_apple_different_view`

### Final reconstructed dataset

- Original retained before review: 4,986
- Unique excluded images: 22
- Retained: 4,964
- Train: 3,475
- Validation: 745
- Test: 744
- Exact duplicate leakage across splits: 0
- Missing/unreadable files: 0
- All four classes present in every split: yes
- Source dataset modified: no

### Generated files

- `src/phase4_2_reconstruct.py`
- `tests/test_phase4_2.py`
- `dataset_final/train/`, `dataset_final/val/`, `dataset_final/test/`
- `results/phase4_2/PHASE4_2_FINAL_DATASET_REPORT.md`
- `results/phase4_2/final_dataset_manifest.csv`
- `results/phase4_2/final_dataset_summary.json`
- `results/phase4_2/exclusion_audit.csv`
- `results/phase4_2/validation_report.json`

### Readiness

The final dataset is ready with warnings for YOLO11 **single-image
classification preparation**, subject to installing the supported ML
environment. It is not a detection dataset and true four-view evaluation
remains unsupported.

## Phase 5 - ML environment setup

**Status:** Environment audit complete; installation and training not started

### Audit results

- Only Python 3.14.7 is installed; Python 3.11 is unavailable.
- pip 26.2.1 is available.
- PyTorch and Ultralytics are not installed.
- `nvidia-smi` failed with an insufficient-permissions error; GPU/CUDA status
  is unknown.
- RAM: 15.71 GB total, 4.42 GB free at audit time.
- Disk: C: 85.35 GB free; D: 201.42 GB free.
- Final dataset verified: 4,964 files, split 3,475/745/744.
- Source dataset remains present with 5,566 files and was not modified.

### Configuration findings

- `requirements.txt` documents Python 3.11 and required dependencies.
- `config/settings.yaml` currently says `yolo11n.pt` and image size 640;
  classification training requires `yolo11n-cls.pt` and the requested
  baseline image size 224. These were not changed during the audit.

### Generated file

- `results/phase5/ENVIRONMENT_AUDIT_REPORT.md`

### Phase 5.1 preparation

- Python 3.11 remains unavailable; Python 3.14 was preserved.
- Windows WMI identified Intel integrated graphics only:
  `Intel(R) Graphics`, driver 31.0.101.5333.
- No NVIDIA GPU was identified; CPU-first PyTorch setup is recommended.
- Updated `config/settings.yaml` for classification:
  `yolo11n-cls.pt`, 224px, task `classify`, four explicit class names, and
  `dataset_final`.
- Added `results/phase5/PHASE5_1_SETUP_REPORT.md`.
- No packages were installed, no dataset files were changed, and training was
  not started.

### Decision

Install Python 3.11 alongside Python 3.14, create `.venv`, verify the runtime,
then select a CPU PyTorch build from the official selector and install project
requirements. Rerun the readiness gate before Phase 5 training.

## Phase 5.2 - ML dependency installation and configuration

**Status:** Complete - CPU environment verified; training not started

### Completed

- Verified project `.venv` uses Python 3.11.9 and pip 26.2.2.
- Retained compatible CPU-only PyTorch packages:
  - PyTorch 2.15.0+cpu
  - Torchvision 0.29.2+cpu
  - Torchaudio 2.11.1+cpu
- Installed and verified project requirements inside `.venv`.
- Verified Ultralytics 8.4.197.
- Verified CPU tensor operations and `torch.cuda.is_available() == False`.
- Loaded `yolo11n-cls.pt` successfully for classification without training.
- `pip check`: no broken requirements.
- Updated `requirements.txt` with the official CPU PyTorch installation command.
- Generated `requirements-installed.txt` from the actual virtual environment.
- Confirmed `config/settings.yaml` uses `yolo11n-cls.pt`, task `classify`,
  image size 224, `dataset_final`, and the four explicit classes.
- Project tests: 13 passed.
- Final and source datasets were not modified.

### Generated files

- `requirements-installed.txt`
- `results/phase5/PHASE5_2_DEPENDENCY_INSTALLATION_REPORT.md`

### Warnings and next phase

- The host uses Intel integrated graphics; CUDA is unavailable and CPU
  training is expected to be slow.
- No training was started.
- Phase 5.3 may implement the training script, but must perform a final
  readiness check before running any experiment.

## Phase 5.3 - Training pipeline implementation

**Status:** Complete - readiness passed; training not started

### Completed

- Added configurable YOLO11 classification training pipeline in
  `src/train.py`.
- Added reusable configuration, dataset, dependency, and device utilities in
  `src/utils/training_utils.py`.
- Added metric helpers in `src/utils/metrics.py`.
- Added separate validation/test evaluation entry points in `src/evaluate.py`.
- Added CPU-safe configuration with zero data-loader workers.
- Updated settings with `workers: 0`.
- Added tests for configuration, dataset validation, argument generation,
  checkpoint/run handling, and metric calculations.
- Readiness gate passed on Python 3.11.9 with the approved 4,964-image final
  dataset.
- Tests: 17 passed.
- No training, smoke test, or test-set evaluation was run.
- No dataset files or checkpoints were created.

### Generated report

- `results/phase5/PHASE5_3_TRAINING_PIPELINE_REPORT.md`

### Phase 5.4 prerequisites

The project is ready for a separately approved training experiment. Before
that experiment, rerun `src/train.py --check-only`; then use the explicit
`--train` command. Keep final test evaluation separate from training and
validation selection.

## Phase 5.4 - CPU smoke test and pipeline verification

**Status:** Complete - PASS WITH WARNINGS

### Completed

- Passed the smoke-dataset readiness gate.
- Created an isolated 48-image smoke dataset from final train/validation data
  only: 32 train images and 16 validation images, with all four classes in
  both splits.
- Completed exactly one CPU training epoch using `yolo11n-cls.pt`, image size
  224, batch size 4, workers 0, and seed 42.
- Verified `best.pt` and `last.pt` load with Ultralytics.
- Verified inference on a smoke validation image.
- Confirmed the final test split was not used and `dataset_final/` was not
  modified.
- Full test suite: 17 passed.
- Added `results/phase5/PHASE5_4_CPU_SMOKE_TEST_REPORT.md`.
- Updated training argument construction to resolve future Ultralytics output
  project paths absolutely.

### Warning

Ultralytics placed the completed run under
`runs/classify/runs/appleguard/smoke_tests/cpu_1epoch/` because the original
project argument was relative. This is documented in the phase report and was
corrected for future runs. Full training remains blocked pending explicit user
approval.

## Phase 6 - Full YOLO11 classification training

**Status:** Complete - SUCCESS

### Completed

- Added the dedicated configuration
  `config/phase6_full_training_v1.yaml` with the approved 30-epoch,
  batch-8, CPU, workers-0, patience-10 settings.
- Passed both the standard and dedicated Phase 6 readiness gates.
- Completed one full CPU training experiment in
  `runs/appleguard/classification/full_training_v1/`.
- Completed all 30 epochs in 5,053.38 seconds; early stopping was not
  triggered.
- Verified `best.pt` and `last.pt` load as Ultralytics classification models.
- Preserved training history, effective configuration, summary, plots, and
  checkpoints.
- Confirmed dataset counts remained 3,475 train, 745 validation, and 744
  test images.
- Confirmed the final test split was not used.
- Regression tests: 17 passed.
- Added `results/phase6/PHASE6_FULL_TRAINING_REPORT.md`.

### Validation observations

- Final recorded validation top-1: 0.99597.
- Final recorded validation top-5: 1.0.
- Best recorded validation top-1 in the history: 0.99866.
- These are validation-only observations and are not final test results.

### Limitations and next step

The model is a four-class image classifier, not an object detector, and
four-view same-apple recognition remains unvalidated. Phase 7 requires a
separate approved evaluation on the held-out test split.

## Phase 7 - Final held-out test evaluation

**Status:** Complete - SUCCESS WITH WARNINGS

### Completed

- Verified Python 3.11.9, the approved `best.pt` checkpoint, checkpoint task
  `classify`, class ordering, and 744 readable test images.
- Evaluated the held-out test split with CPU inference at image size 224,
  batch size 8, and workers 0.
- Measured 738 correct and 6 incorrect predictions.
- Test top-1/accuracy: 0.991935.
- Test macro precision/recall/F1: 0.992141 / 0.991611 / 0.991827.
- Test weighted precision/recall/F1: 0.991997 / 0.991935 / 0.991918.
- Test top-5 accuracy: 1.0; this is trivially complete because there are four
  classes.
- Generated the required Phase 7 reports, prediction CSV, classification
  reports, summary JSON, and confusion matrix artifacts.
- Final dataset counts remained 3,475 train, 745 validation, and 744 test.
- Regression tests: 17 passed.
- Added `results/phase7/PHASE7_FINAL_EVALUATION_REPORT.md`.

### Warning

An initial prediction pass reached the output-writing stage but failed on a
report-formatting bug. The output-only bug was corrected and the final
artifact-producing evaluation completed successfully. No test result was used
for model or threshold changes. The model remains a classifier, not an object
detector, and four-view recognition remains unvalidated.

## Phase 8 - Model deployment and prediction application

**Status:** Complete - verified

### Completed

- Added the Streamlit application in `app.py` with Home, Disease Prediction,
  Prediction History, and About Model pages.
- Added cached loading of the approved Phase 6 `best.pt` checkpoint.
- Added CPU-safe inference with optional CUDA selection only when available.
- Added JPG/JPEG/PNG validation, 10 MB size limit, corrupted-image handling,
  all-class probabilities, confidence, timestamps, and device display.
- Added local metadata-only per-session SQLite history with duplicate
  prevention and clear-history support.
- Added agricultural information content with non-prescriptive guidance.
- Added `APP_README.md`, `requirements-app.txt`, styles, application utilities,
  and Phase 8 tests.
- Application and regression tests: 22 passed.
- Checkpoint inference verification: passed on CPU.
- Streamlit health endpoint verification: HTTP 200 (`ok`).
- Dataset image count remains 4,964; Phase 7 artifacts remain present.
- Added `results/phase8/PHASE8_IMPLEMENTATION_REPORT.md`.

### Limitations

The application is an image classifier, not an object-detection system, and
does not draw disease-region boxes. Four-view same-apple recognition is not
validated. Prediction confidence is not a guaranteed diagnosis.

## Phase 9 - Production readiness, security, performance, and deployment

**Status:** Complete - SUCCESS WITH WARNINGS

### Completed

- Recorded the initial repository/application audit.
- Hardened upload size enforcement, image decoding, pixel limits, filename
  handling, CSV formula safety, history schema validation, and error handling.
- Added session inference deduplication, loading feedback, responsive styling,
  and clearer upload guidance.
- Benchmarked CPU startup, checkpoint loading, warm-up, and repeated inference.
- Diagnosed NVIDIA hardware separately; `nvidia-smi` sees an RTX 3050, but the
  installed PyTorch build is CPU-only and CUDA inference was not claimed.
- Added deployment, user, technical, changelog, security, performance, GPU,
  and final Phase 9 documentation.
- Tests: 28 passed; `pip check` passed; Streamlit health returned HTTP 200.
- Verified the approved checkpoint hash remained
  `0C2951D9DD3B1CF32EAF75BC140CECA7144D150AD6F9DE84CD0671242482AD4C`.
- Confirmed dataset image count remains 4,964 and Phase 7 artifacts remain
  unchanged.

### Warning

Prediction history is isolated per Streamlit session in SQLite. External
authentication remains outside scope and is required before public deployment.

## Phase 10 - NVIDIA GPU acceleration and inference optimization

**Status:** Complete - SUCCESS

### Completed

- Audited NVIDIA hardware: RTX 3050 Laptop GPU, 4 GB VRAM, driver 591.97,
  driver-reported CUDA compatibility 13.1.
- Preserved `.venv` CPU environment and created separate `.venv-gpu`.
- Installed official CUDA 12.6 PyTorch wheels:
  torch 2.14.1+cu126, torchvision 0.29.1+cu126, torchaudio 2.11.0+cu126.
- Verified `torch.cuda.is_available()` is true in `.venv-gpu`.
- Loaded the approved `best.pt` checkpoint and verified real CUDA inference
  on `cuda:0` with four finite probabilities and matching class mapping.
- Confirmed CPU/GPU predictions agree on four identical non-test images within
  a maximum probability difference below 2e-5.
- Measured CUDA throughput at 258.22 images/s versus CPU 145.18 images/s in
  the controlled batch benchmark.
- Added CUDA failure fallback to CPU in the application.
- Verified Streamlit health in both CPU and GPU environments.
- CPU tests: 28 passed; GPU tests: 28 passed.
- Added `GPU_SETUP_GUIDE.md`, `requirements-gpu.txt`, and Phase 10 reports.

### Limitations

The application remains an image classifier, not an object detector, and
four-view same-apple recognition is not validated. Public deployment was not
performed.
## Phase 11 - UI Redesign and Explainable Prediction Experience

Status: **SUCCESS WITH WARNINGS**

- Redesigned the Streamlit application with a responsive professional light
  theme and sidebar navigation.
- Added dashboard, explicit image analysis, confidence-band interpretation,
  sorted probability comparison, output-based explanations, low-confidence
  guidance, and history filters/confirmation.
- Preserved upload validation, session-isolated SQLite history, model caching,
  CPU fallback, and verified CUDA support.
- Added focused explanation tests and completed full regression testing:
  35 tests passed in both CPU and GPU environments.
- CPU and GPU Streamlit health endpoints returned HTTP 200.
- See `results/phase11/PHASE11_FINAL_REPORT.md`.
## Phase 11.1 - Premium UI Transformation

Status: **SUCCESS WITH WARNINGS**

- Rebuilt the Streamlit page structure as a premium light agricultural AI
  platform with six navigation pages and a breadcrumb/system-status shell.
- Added richer dashboard KPIs, workflow, category showcase, model snapshot,
  upload workspace, confidence meter, probability distribution, explanation
  cards, disease library, performance analytics, and project context.
- Added original CSS/HTML visual branding without fabricating disease evidence.
- Added Streamlit configuration enforcing the 10 MB upload limit.
- CPU and GPU full suites: 35 tests passed in each environment.
- CPU and GPU Streamlit health endpoints: HTTP 200.
- See `results/phase11/PHASE11_1_PREMIUM_UI_REPORT.md`.
