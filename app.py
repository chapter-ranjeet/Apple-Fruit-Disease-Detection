from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils.explanation import build_explanation, confidence_band, sorted_probabilities
from app.utils.history import append_prediction, clear_history, read_history
from app.utils.inference import CLASS_NAMES, DISPLAY_NAMES, available_devices, load_model, predict_image
from app.utils.validation import read_uploaded_file, safe_filename

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "runs" / "appleguard" / "classification" / "full_training_v1" / "weights" / "best.pt"
STYLE_PATH = ROOT / "assets" / "styles.css"
CONFUSION_MATRIX = ROOT / "results" / "phase7" / "confusion_matrix.png"

INFO = {
    "Blotch_Apple": {
        "title": "Apple Blotch",
        "description": "A supported apple category associated with blotch-like disease symptoms.",
        "reference": "General references may describe dark or irregular marks on fruit or foliage. This is not an observation made by the model.",
        "guidance": "Use orchard sanitation and local integrated disease-management guidance; consult an agricultural expert.",
        "accent": "blue",
        "symbol": "B",
    },
    "Normal_Apple": {
        "title": "Normal Apple",
        "description": "The supported category for an image with no disease category favored by the model.",
        "reference": "A healthy reference can include intact, uniform-looking fruit, but image quality and context affect classification.",
        "guidance": "Continue routine monitoring, sanitation, balanced irrigation, and expert consultation when concerns persist.",
        "accent": "green",
        "symbol": "N",
    },
    "Rot_Apple": {
        "title": "Apple Rot",
        "description": "A supported category associated with decay-related apple symptoms.",
        "reference": "General references may describe soft, discolored, or decaying tissue. This is not an observation made by the model.",
        "guidance": "Reduce injury and moisture exposure, remove decayed material safely, and consult an agricultural expert.",
        "accent": "orange",
        "symbol": "R",
    },
    "Scab_Apple": {
        "title": "Apple Scab",
        "description": "A supported category associated with scab-like apple symptoms.",
        "reference": "General references may describe olive, brown, or dark lesions on fruit or leaves. This is not an observation made by the model.",
        "guidance": "Use sanitation and local expert guidance; do not apply treatments based only on this prediction.",
        "accent": "teal",
        "symbol": "S",
    },
}

PAGE_META = {
    "Dashboard": ("Dashboard", "A clear overview of your apple health analysis workspace."),
    "Disease Detection": ("Disease Detection", "Classify a complete apple image across four supported categories."),
    "Prediction History": ("Prediction History", "Review private prediction metadata from this session."),
    "Disease Library": ("Disease Library", "Explore general reference information for each category."),
    "Model Performance": ("Model Performance", "Review historical evaluation and device benchmarks."),
    "About Project": ("About Project", "Understand the project scope, model, and limitations."),
}


@st.cache_resource(show_spinner=False)
def get_model(device: str):
    return load_model(CHECKPOINT, device)


def get_history_path() -> Path:
    session_id = st.session_state.setdefault("history_session_id", secrets.token_hex(16))
    return ROOT / "results" / "phase9" / "history" / f"{session_id}.sqlite3"


def render_header(page: str) -> None:
    title, subtitle = PAGE_META[page]
    st.markdown(
        f'<div class="top-header"><div><div class="breadcrumb">APPLE-FRUIT-DISEASE-DETECTION <span>/</span> {title.upper()}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div><div class="system-pill"><span class="status-dot"></span> System operational</div></div>',
        unsafe_allow_html=True,
    )


def render_visual_tile(label: str, accent: str = "blue", symbol: str = "A") -> None:
    st.markdown(
        f'<div class="visual-tile {accent}"><div class="visual-orbit"></div><div class="visual-fruit">{symbol}</div><span>{label}</span></div>',
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown(
        '<section class="premium-hero"><div class="hero-copy"><span class="eyebrow">AI-POWERED AGRICULTURE</span>'
        '<h2>Smarter apple health<br><em>analysis starts here.</em></h2>'
        '<p>Understand apple images with a verified YOLO11 classification model, transparent probabilities, and practical context for your next decision.</p>'
        '<div class="hero-actions"><a class="fake-primary" href="#analyze-apple-image">Analyze Apple Image <span>→</span></a><span class="hero-note">Whole-image classification · 224 × 224</span></div>'
        '</div><div class="hero-visual"><div class="hero-sun"></div><div class="hero-leaf leaf-one"></div><div class="hero-leaf leaf-two"></div><div class="hero-apple">A</div>'
        '<div class="float-card float-one"><strong>4</strong><span>Supported<br>categories</span></div><div class="float-card float-two"><strong>YOLO11</strong><span>AI model<br>architecture</span></div><div class="float-card float-three"><strong>CPU / GPU</strong><span>Inference<br>ready</span></div></div></section>',
        unsafe_allow_html=True,
    )
    st.markdown('<div id="analyze-apple-image"></div>', unsafe_allow_html=True)
    kpis = st.columns(4)
    kpi_data = [
        ("◈", "4", "Supported categories", "Blotch, Normal, Rot, Scab", "blue"),
        ("▦", "4,964", "Approved images", "Leakage-safe final dataset", "green"),
        ("↗", "99.19%", "Held-out accuracy", "Historical Phase 7 result", "purple"),
        ("◉", "GPU Ready" if "cuda:0" in available_devices() else "CPU Ready", "Inference availability", "Safe automatic fallback", "orange"),
    ]
    for column, (icon, value, label, detail, accent) in zip(kpis, kpi_data):
        column.markdown(f'<div class="kpi-card {accent}"><div class="kpi-icon">{icon}</div><strong>{value}</strong><span>{label}</span><small>{detail}</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="eyebrow">A SIMPLE WORKFLOW</span><h2>From image to understanding</h2><p>Designed to keep the model result clear, contextual, and easy to review.</p></div>', unsafe_allow_html=True)
    steps = st.columns(4)
    for column, number, icon, title, text in zip(
        steps,
        ("01", "02", "03", "04"),
        ("↑", "✦", "≡", "◌"),
        ("Upload image", "AI analysis", "Review result", "Understand context"),
        ("Choose a clear JPG, JPEG, or PNG.", "The complete image is classified by YOLO11.", "Compare confidence and all four outputs.", "Read general guidance and limitations."),
    ):
        column.markdown(f'<div class="workflow-card"><div class="workflow-icon">{icon}</div><small>{number}</small><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="eyebrow">SUPPORTED CATEGORIES</span><h2>Know the categories</h2><p>General reference information is kept separate from what the model observed.</p></div>', unsafe_allow_html=True)
    cards = st.columns(4)
    for column, class_name in zip(cards, CLASS_NAMES):
        info = INFO[class_name]
        column.markdown(f'<div class="disease-card"><div class="mini-visual {info["accent"]}">{info["symbol"]}</div><span class="card-label">{info["title"]}</span><p>{info["description"]}</p><small>Explore in Disease Library →</small></div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, .8])
    with left:
        st.markdown('<div class="insight-card"><span class="eyebrow">MODEL SNAPSHOT</span><h3>Built for transparent review</h3><p>Historical evaluation reached <strong>99.19% accuracy</strong> on 744 held-out test images. This is an offline project result—not a guarantee for every uploaded image.</p><div class="insight-metrics"><span><strong>99.18%</strong><small>Macro F1</small></span><span><strong>224</strong><small>Input pixels</small></span><span><strong>4</strong><small>Classes</small></span></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="notice premium-notice"><strong>Use the result responsibly</strong><p>This educational tool does not diagnose, locate disease, measure severity, or replace an agricultural expert.</p></div>', unsafe_allow_html=True)


def render_prediction() -> None:
    st.markdown('<div class="prediction-intro"><span class="eyebrow">ANALYSIS WORKSPACE</span><h2>Upload an apple image</h2><p>Get a transparent four-category classification with all model outputs visible.</p></div>', unsafe_allow_html=True)
    device = st.selectbox("Inference device", available_devices(), index=0)
    upload_key = f"upload_{st.session_state.get('upload_widget_version', 0)}"
    upload_col, guide_col = st.columns([1.45, .55])
    with upload_col:
        uploaded = st.file_uploader("Drop your image here or browse files", type=["jpg", "jpeg", "png"], key=upload_key, help="Images are validated and processed in memory. Uploaded image bytes are not saved.")
        st.caption("JPG, JPEG, or PNG · Maximum 10 MB · One clear apple image works best")
    with guide_col:
        st.markdown('<div class="capture-card"><div class="capture-icon">◎</div><h3>Capture a useful image</h3><p>Use even lighting, keep the subject in focus, and avoid heavy blur or obstruction.</p><small>Tip: a clear, close subject helps the model make a more useful comparison.</small></div>', unsafe_allow_html=True)
    if uploaded is None:
        st.markdown('<div class="upload-empty"><div class="upload-symbol">↑</div><h3>Ready for your image</h3><p>Drag and drop an apple image above to begin.</p></div>', unsafe_allow_html=True)
        return
    try:
        data, image = read_uploaded_file(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    display_filename = safe_filename(uploaded.name)
    preview_col, action_col = st.columns([4, 1])
    with preview_col:
        st.image(image, caption=display_filename, use_container_width=True)
    with action_col:
        if st.button("Replace image", type="secondary"):
            st.session_state["upload_widget_version"] = st.session_state.get("upload_widget_version", 0) + 1
            st.session_state.pop("prediction_cache", None)
            st.rerun()
    analyze = st.button("Analyze image", type="primary", use_container_width=True)
    signature = hashlib.sha256(data + device.encode()).hexdigest()
    cached_prediction = st.session_state.get("prediction_cache", {})
    if not analyze and signature not in cached_prediction:
        st.info("Select **Analyze image** to run the model.")
        return
    try:
        if signature in cached_prediction:
            prediction = cached_prediction[signature]
        else:
            with st.spinner("Analyzing image with the verified model..."):
                model = get_model(device)
                prediction = predict_image(model, image, device)
            st.session_state.setdefault("prediction_cache", {})[signature] = prediction
    except (FileNotFoundError, RuntimeError, ValueError):
        st.error("The prediction could not be completed. Check the model installation and image, then try again.")
        return
    band, band_guidance = confidence_band(prediction["confidence"])
    info = INFO[prediction["class_name"]]
    st.markdown('<div class="section-title result-heading"><span class="eyebrow">ANALYSIS COMPLETE</span><h2>Your AI result</h2></div>', unsafe_allow_html=True)
    result_left, result_right = st.columns([1, 1])
    with result_left:
        st.markdown(f'<div class="result-panel"><div class="result-symbol {info["accent"]}">{info["symbol"]}</div><span class="eyebrow">PREDICTED CATEGORY</span><h2>{prediction["display_name"]}</h2><div class="confidence-line"><strong>{prediction["confidence"]:.2%}</strong><span>{band}</span></div><div class="meter"><div style="width:{prediction["confidence"] * 100:.2f}%"></div></div><p>{band_guidance}</p><small>Actual inference device: {prediction["device"]}</small></div>', unsafe_allow_html=True)
    with result_right:
        st.markdown('<div class="probability-panel"><span class="eyebrow">PROBABILITY DISTRIBUTION</span><h3>How the categories compare</h3>', unsafe_allow_html=True)
        for class_name, probability in sorted_probabilities(prediction["probabilities"]):
            is_top = class_name == prediction["class_name"]
            label = f'{DISPLAY_NAMES[class_name]} <b class="top-tag">TOP</b>' if is_top else DISPLAY_NAMES[class_name]
            st.markdown(f'<div class="probability-row"><div><span>{label}</span><strong>{probability:.2%}</strong></div><div class="probability-track"><i class="{"top" if is_top else ""}" style="width:{probability * 100:.2f}%"></i></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="explain-card"><span class="eyebrow">EXPLAINABLE RESULT</span><h2>Understanding your AI result</h2>', unsafe_allow_html=True)
    explanation_cols = st.columns(2)
    with explanation_cols[0]:
        st.markdown(f"**What this category means**  \n{info['description']}  \n\n**What the output tells us**  \n{build_explanation(prediction['class_name'], prediction['probabilities'])}")
    with explanation_cols[1]:
        st.markdown(f"**What this result cannot establish**  \nIt does not confirm a diagnosis, identify a disease location, or measure severity.  \n\n**Recommended next steps**  \nCapture another clear image, compare with trusted agricultural resources, and consult an expert if concerns persist.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="notice">The confidence bands are UI interpretation labels only: high (85%+), moderate (65–84.99%), and low (below 65%). They are not validated diagnostic thresholds.</div>', unsafe_allow_html=True)
    if prediction["confidence"] < 0.65:
        st.warning("Low model confidence: try a clearer, well-lit, in-focus image before interpreting the result.")
    history_signature = hashlib.sha256(data + prediction["class_name"].encode()).hexdigest()
    try:
        append_prediction(get_history_path(), {"filename": display_filename, "predicted_class": prediction["class_name"], "confidence": prediction["confidence"], "device": prediction["device"]}, history_signature)
    except (OSError, UnicodeError, ValueError):
        st.warning("Prediction completed, but history could not be saved.")
    action_one, action_two = st.columns(2)
    with action_one:
        if st.button("Analyze another image", use_container_width=True):
            st.session_state["upload_widget_version"] = st.session_state.get("upload_widget_version", 0) + 1
            st.session_state.pop("prediction_cache", None)
            st.rerun()
    with action_two:
        st.caption(f"Prediction timestamp: {datetime.now(timezone.utc).isoformat()}")


def render_history() -> None:
    st.markdown('<div class="page-heading"><span class="eyebrow">SESSION RECORDS</span><h2>Your prediction activity</h2><p>Metadata is isolated to this Streamlit session. Uploaded images and thumbnails are not stored.</p></div>', unsafe_allow_html=True)
    history_path = get_history_path()
    try:
        records = read_history(history_path)
    except (OSError, UnicodeDecodeError, ValueError):
        st.error("Prediction history could not be read safely.")
        records = []
    filter_col, search_col, clear_col = st.columns([1, 2, 1])
    category = filter_col.selectbox("Category", ["All categories"] + [DISPLAY_NAMES[name] for name in CLASS_NAMES])
    search = search_col.text_input("Search filename", placeholder="Search this session")
    if clear_col.button("Clear history", type="secondary"):
        st.session_state["confirm_clear_history"] = True
    if st.session_state.get("confirm_clear_history"):
        st.warning("Clear all prediction metadata from this session?")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm clear", type="primary"):
            try:
                clear_history(history_path)
            except OSError:
                st.error("Prediction history could not be cleared.")
            else:
                st.session_state["confirm_clear_history"] = False
                st.rerun()
        if cancel_col.button("Cancel"):
            st.session_state["confirm_clear_history"] = False
            st.rerun()
    filtered = [row for row in records if (category == "All categories" or DISPLAY_NAMES.get(row["predicted_class"], row["predicted_class"]) == category) and search.lower() in row["filename"].lower()]
    if filtered:
        st.markdown(f'<div class="history-summary"><strong>{len(filtered)}</strong> prediction record{"s" if len(filtered) != 1 else ""} in this session</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)
    elif records:
        st.info("No history records match the current filters.")
    else:
        st.markdown('<div class="empty-state"><div class="empty-symbol">≡</div><h3>No predictions yet</h3><p>Run an image analysis to create a private session record.</p></div>', unsafe_allow_html=True)


def render_library() -> None:
    st.markdown('<div class="page-heading"><span class="eyebrow">REFERENCE CENTER</span><h2>Disease library</h2><p>General agricultural context for the four categories supported by this classifier.</p></div>', unsafe_allow_html=True)
    for row_start in range(0, len(CLASS_NAMES), 2):
        columns = st.columns(2)
        for column, class_name in zip(columns, CLASS_NAMES[row_start:row_start + 2]):
            info = INFO[class_name]
            with column:
                st.markdown(f'<div class="library-card"><div class="library-head"><div class="large-visual {info["accent"]}">{info["symbol"]}</div><div><span class="eyebrow">CATEGORY</span><h3>{info["title"]}</h3></div></div><p>{info["description"]}</p></div>', unsafe_allow_html=True)
                with st.expander(f"Explore {info['title']}"):
                    st.write("**General reference characteristics**")
                    st.write(info["reference"])
                    st.write("**Informational guidance**")
                    st.write(info["guidance"])
                    st.write("Consult an agricultural expert when symptoms persist or decisions may affect crop health.")


def render_performance() -> None:
    st.markdown('<div class="page-heading"><span class="eyebrow">MODEL ANALYTICS</span><h2>Model performance</h2><p>Historical project evaluation and controlled device benchmarks—not live prediction guarantees.</p></div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    for column, value, label in zip(metrics, ("99.1935%", "99.2141%", "99.1611%", "99.1827%"), ("Test accuracy", "Macro precision", "Macro recall", "Macro F1")):
        column.markdown(f'<div class="performance-metric"><strong>{value}</strong><span>{label}</span><small>Phase 7 held-out test</small></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="data-card"><span class="eyebrow">DATASET SPLIT</span><h3>4,964 approved images</h3>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Images": [3475, 745, 744]}, index=["Train", "Validation", "Test"]), height=220)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="data-card"><span class="eyebrow">DEVICE BENCHMARK</span><h3>Controlled Phase 10 comparison</h3><div class="benchmark-row"><span>CPU throughput</span><strong>145.18 images/s</strong></div><div class="benchmark-row"><span>CUDA throughput</span><strong>258.22 images/s</strong></div><div class="benchmark-row"><span>CUDA speedup</span><strong>1.78×</strong></div><small>Four-image batches, 30 repetitions, warm-up excluded. Hardware and batch specific.</small></div>', unsafe_allow_html=True)
    if CONFUSION_MATRIX.exists():
        st.markdown('<div class="data-card"><span class="eyebrow">CONFUSION MATRIX</span><h3>Phase 7 held-out evaluation</h3><p>Existing project artifact; not regenerated or tuned for this interface.</p>', unsafe_allow_html=True)
        st.image(str(CONFUSION_MATRIX), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Metrics are historical held-out test results. They should not be read as a guarantee for future images.")


def render_about() -> None:
    st.markdown('<div class="page-heading"><span class="eyebrow">PROJECT CONTEXT</span><h2>About this project</h2><p>A transparent educational platform for whole-image apple category classification.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, .8])
    with left:
        st.markdown('<div class="about-card"><span class="eyebrow">THE MODEL</span><h3>YOLO11 Classification</h3><p>The approved Phase 6 checkpoint receives a 224 × 224 image and returns probabilities for four classes: Apple Blotch, Normal Apple, Apple Rot, and Apple Scab.</p><div class="tag-row"><span>Python 3.11</span><span>Streamlit</span><span>PyTorch</span><span>Ultralytics</span></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="about-card"><span class="eyebrow">SYSTEM STATUS</span><h3>Ready for private use</h3><p>CPU support is always preserved. CUDA is offered only when available and falls back safely if initialization or inference fails.</p><div class="status-line"><span class="status-dot"></span> Model checkpoint available</div></div>', unsafe_allow_html=True)
    st.markdown("### Boundaries that matter")
    limits = st.columns(3)
    for column, title, text in zip(limits, ("Classification only", "Not a diagnosis", "No multi-view claim"), ("The app classifies the complete image and does not draw bounding boxes or locate disease.", "Confidence is a relative model score and cannot confirm disease or severity.", "The dataset does not validate recognition of four views of the same physical apple.")):
        column.markdown(f'<div class="limit-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">This is an experimental image-classification aid for education and review. Consult qualified agricultural expertise for real-world decisions.</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Apple-Fruit-Disease-Detection", page_icon="A", layout="wide", initial_sidebar_state="expanded")
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-brand"><div class="brand-mark">A</div><div><strong>Apple Health<br>Intelligence</strong><small>Apple-Fruit-Disease-Detection</small></div></div>', unsafe_allow_html=True)
    options = ["Dashboard", "Disease Detection", "Prediction History", "Disease Library", "Model Performance", "About Project"]
    page = st.sidebar.radio("Navigation", options, format_func=lambda value: {"Dashboard": "⌂   Dashboard", "Disease Detection": "◉   Disease Detection", "Prediction History": "▤   Prediction History", "Disease Library": "◇   Disease Library", "Model Performance": "▥   Model Performance", "About Project": "ⓘ   About Project"}[value], label_visibility="collapsed")
    st.sidebar.markdown("---")
    device_label = "CUDA GPU" if "cuda:0" in available_devices() else "CPU"
    st.sidebar.markdown(f'<div class="sidebar-status"><div class="status-line"><span class="status-dot"></span><strong>Model loaded</strong></div><span>YOLO11 Classification</span><br><span>Inference: {device_label}</span><br><span>Application version: 11.1</span></div>', unsafe_allow_html=True)
    render_header(page)
    if page == "Dashboard":
        render_home()
    elif page == "Disease Detection":
        render_prediction()
    elif page == "Prediction History":
        render_history()
    elif page == "Disease Library":
        render_library()
    elif page == "Model Performance":
        render_performance()
    else:
        render_about()


if __name__ == "__main__":
    main()
