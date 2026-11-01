# Changelog

## Phase 11.1

- Transformed the Streamlit shell into a premium responsive light interface.
- Added six-page navigation: Dashboard, Disease Detection, Prediction History,
  Disease Library, Model Performance, and About Project.
- Added original CSS agricultural visual treatment, richer KPI/workflow cards,
  upload workspace, confidence meter, result panels, library cards, and
  real-artifact performance analytics.
- Added `.streamlit/config.toml` with a 10 MB Streamlit upload limit and light
  theme defaults.
- Preserved verified model inference, security validation, session history,
  CPU fallback, CUDA support, and classifier-only scope.

## Phase 11

- Redesigned the Streamlit interface with a professional responsive light theme.
- Added dashboard workflow cards, supported-category cards, model status, and
  clearer agricultural-use notices.
- Added explicit image analysis, remove/replace behavior, confidence bands,
  sorted four-class probability bars, output-based explanations, low-confidence
  guidance, and next-step guidance.
- Added history search, category filtering, and clear-history confirmation.
- Added reusable explanation utilities and focused tests.
- Preserved upload validation, SQLite session isolation, CPU fallback, CUDA
  support, model caching, and classifier-only scope.

## Phase 9

- Hardened upload reads, image decoding, pixel limits, and filename handling.
- Added CSV formula protection, history schema validation, and in-process
  duplicate-write locking.
- Added bounded user-facing errors and session-level inference deduplication.
- Improved responsive styling, upload guidance, loading feedback, and empty
  states.
- Added security, performance, GPU compatibility, deployment, user, and
  technical documentation.
- Added Phase 9 security, fallback, history, and checkpoint tests.

## Phase 8

- Added the Streamlit classification application with four navigation pages,
  cached model loading, prediction history, and model information.
