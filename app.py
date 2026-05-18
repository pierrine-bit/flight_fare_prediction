"""
app.py — Streamlit Flight Fare Prediction UI.

Usage:  streamlit run app.py

Requires outputs/best_model.pkl and outputs/preprocessor.pkl (run main.py first).

Design notes
────────────
• Model name is read dynamically from the loaded estimator — never hardcoded.
• @st.cache_resource loads artefacts once per session, not per interaction.
• engineer_features() mirrors the training pipeline so derived features
  (Month, Weekday, Stop_Type) are always computed consistently at inference.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR, LOG_TRANSFORM_TARGET,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)
from src.feature_engineering import engineer_features


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Flight Fare Predictor — Bangladesh",
    page_icon=":airplane:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #6c757d; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #212529; margin-top: 0.2rem; }

    .result-box {
        background: linear-gradient(135deg, #0f3460, #533483);
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-box .fare-label { font-size: 0.85rem; opacity: 0.8; letter-spacing: 0.08em;
                              text-transform: uppercase; }
    .result-box .fare-value { font-size: 3rem; font-weight: 800; margin: 0.3rem 0; }
    .result-box .fare-range { font-size: 0.8rem; opacity: 0.7; }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { font-weight: 600; font-size: 0.85rem; }

    .section-title {
        font-size: 0.7rem; font-weight: 700; color: #6c757d;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.5rem; border-bottom: 1px solid #e9ecef;
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Artefact loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    """
    Load model, preprocessor, and optional metrics from disk.
    Calls st.stop() with a user-friendly error if required artefacts are missing.
    """
    model_path   = MODELS_DIR  / "best_model.pkl"
    prep_path    = MODELS_DIR  / "preprocessor.pkl"
    metrics_path = REPORTS_DIR / "model_metrics.csv"

    if not model_path.exists() or not prep_path.exists():
        st.error(
            "**Model artefacts not found.**  "
            "Run `python main.py` from the project root to train and save them."
        )
        st.stop()

    model        = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    metrics = None
    if metrics_path.exists():
        try:
            df_m    = pd.read_csv(metrics_path, index_col=0)
            best    = df_m.index[0]
            metrics = {
                "r2":   df_m.loc[best, "CV R² Mean"] if "CV R² Mean" in df_m.columns else None,
                "mae":  df_m.loc[best, "MAE (BDT)"]  if "MAE (BDT)"  in df_m.columns else None,
                "rmse": df_m.loc[best, "RMSE (BDT)"] if "RMSE (BDT)" in df_m.columns else None,
            }
        except Exception:
            pass

    residual_std = None
    residual_std_path = MODELS_DIR / "residual_std.json"
    if residual_std_path.exists():
        try:
            with open(residual_std_path) as f:
                residual_std = json.load(f)["residual_std"]
        except Exception:
            pass

    return model, preprocessor, metrics, residual_std


model, preprocessor, saved_metrics, residual_std = load_artefacts()

# Derive model name dynamically — never hardcode the algorithm
MODEL_NAME = type(model).__name__
MODEL_LABEL = {
    "XGBRegressor":          "XGBoost",
    "RandomForestRegressor": "Random Forest",
    "LGBMRegressor":         "LightGBM",
    "Ridge":                 "Ridge Regression",
    "Lasso":                 "Lasso Regression",
    "LinearRegression":      "Linear Regression",
    "DecisionTreeRegressor": "Decision Tree",
}.get(MODEL_NAME, MODEL_NAME)


# ── Static option lists ───────────────────────────────────────────────────────
# Sourced from the OHE categories fitted during training.
# Unknown values at inference time silently map to all-zero vectors
# (handle_unknown='ignore' in the ColumnTransformer).

AIRLINES = [
    "Air Arabia", "Air Astra", "Air India", "AirAsia",
    "Biman Bangladesh Airlines", "British Airways", "Cathay Pacific",
    "Emirates", "Etihad Airways", "FlyDubai", "Gulf Air", "IndiGo",
    "Kuwait Airways", "Lufthansa", "Malaysian Airlines", "NovoAir",
    "Qatar Airways", "Saudia", "Singapore Airlines", "SriLankan Airlines",
    "Thai Airways", "Turkish Airlines", "US-Bangla Airlines", "Vistara",
]
SOURCES      = ["BZL", "CGP", "CXB", "DAC", "JSR", "RJH", "SPD", "ZYL"]
DESTINATIONS = [
    "BKK", "BZL", "CCU", "CGP", "CXB", "DAC", "DEL", "DOH", "DXB",
    "IST", "JED", "JFK", "JSR", "KUL", "LHR", "RJH", "SIN", "SPD", "YYZ", "ZYL",
]
CLASSES         = ["Economy", "Business", "First"]
AIRCRAFT_TYPES  = ["Airbus A320", "Airbus A350", "Boeing 737", "Boeing 777", "Boeing 787"]
BOOKING_SOURCES = ["Direct Booking", "Online Website", "Travel Agency"]
SEASONS         = ["Regular", "Eid", "Hajj", "Winter Holidays"]
STOP_OPTIONS    = {
    "Non-stop (Direct)": "Direct",
    "1 Stop":            "1 Stop",
    "2 Stops":           "2 Stops",
}


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="main-header">
    <h1>Bangladesh Flight Fare Predictor</h1>
    <p>Powered by <b>{MODEL_LABEL}</b> · trained on 57,000+ Bangladesh flight records · Amalitech Training 2026</p>
</div>
""", unsafe_allow_html=True)

# Model performance metrics row
if saved_metrics:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Model</div>
            <div class="value" style="font-size:1.1rem">{MODEL_LABEL}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        r2_val = f"{saved_metrics['r2']:.4f}" if saved_metrics['r2'] else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="label">CV R²</div>
            <div class="value">{r2_val}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        mae_val = f"BDT {saved_metrics['mae']:,.0f}" if saved_metrics['mae'] else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="label">MAE</div>
            <div class="value" style="font-size:1.1rem">{mae_val}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        rmse_val = f"BDT {saved_metrics['rmse']:,.0f}" if saved_metrics['rmse'] else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="label">RMSE</div>
            <div class="value" style="font-size:1.1rem">{rmse_val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────

st.markdown("### Flight Details")

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<p class="section-title">Route & Airline</p>', unsafe_allow_html=True)
    airline     = st.selectbox("Airline",            AIRLINES,
                               index=AIRLINES.index("Biman Bangladesh Airlines"))
    col_src, col_dst = st.columns(2)
    with col_src:
        source  = st.selectbox("Origin (IATA)",      SOURCES,
                               index=SOURCES.index("DAC"))
    with col_dst:
        destination = st.selectbox("Destination (IATA)", DESTINATIONS,
                                   index=DESTINATIONS.index("DXB"))
    cabin_class = st.selectbox("Cabin Class",        CLASSES)
    stop_label  = st.selectbox("Stopovers",          list(STOP_OPTIONS.keys()))

with right:
    st.markdown('<p class="section-title">Flight & Booking Details</p>', unsafe_allow_html=True)
    aircraft    = st.selectbox("Aircraft Type",      AIRCRAFT_TYPES)
    booking_src = st.selectbox("Booking Channel",    BOOKING_SOURCES)
    season      = st.selectbox("Travel Season",      SEASONS)
    duration    = st.slider("Flight Duration (hrs)", min_value=0.5, max_value=16.0,
                            value=4.0, step=0.5)
    days_before = st.slider("Days Before Departure", min_value=1, max_value=90, value=14)

st.markdown('<p class="section-title" style="margin-top:1rem">Departure Date</p>',
            unsafe_allow_html=True)
dep_date = st.date_input(
    "Departure Date",
    value=pd.Timestamp("today") + pd.Timedelta(days=days_before),
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict Fare", type="primary", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    dep_dt = pd.Timestamp(dep_date)

    # Build a single-row DataFrame matching the training schema
    row = pd.DataFrame([{
        "Airline":               airline,
        "Source":                source,
        "Destination":           destination,
        "Class":                 cabin_class,
        "Aircraft_Type":         aircraft,
        "Booking_Source":        booking_src,
        "Season":                season,
        "Stop_Raw":              STOP_OPTIONS[stop_label],
        "Duration_hrs":          duration,
        "Days_Before_Departure": days_before,
        "Date":                  dep_dt,
    }])

    # Derive Month, Weekday, Stop_Type — identical to the training pipeline
    row = engineer_features(row)

    row_proc = preprocessor.transform(row[CATEGORICAL_FEATURES + NUMERICAL_FEATURES])

    # Predict and reverse log1p transform if applied during training
    pred_log = model.predict(row_proc)[0]
    pred_bdt = float(np.expm1(pred_log)) if LOG_TRANSFORM_TARGET else float(pred_log)

    # 95% prediction interval: pred ± 1.96 × residual_std
    # Falls back to ±MAE if residual_std.json is not yet available
    if residual_std:
        ci_half   = 1.96 * residual_std
        ci_lower  = max(0, pred_bdt - ci_half)
        ci_upper  = pred_bdt + ci_half
        range_str = f"95% Prediction Interval:  BDT {ci_lower:,.0f} — BDT {ci_upper:,.0f}"
        range_note = "(based on test-set residual spread)"
    elif saved_metrics and saved_metrics["mae"]:
        mae_bound = saved_metrics["mae"]
        ci_lower  = max(0, pred_bdt - mae_bound)
        ci_upper  = pred_bdt + mae_bound
        range_str = f"Typical range (±MAE):  BDT {ci_lower:,.0f} — BDT {ci_upper:,.0f}"
        range_note = ""
    else:
        range_str  = ""
        range_note = ""

    st.markdown(f"""
    <div class="result-box">
        <div class="fare-label">Estimated Total Fare</div>
        <div class="fare-value">BDT {pred_bdt:,.0f}</div>
        <div class="fare-range">{range_str}</div>
        <div class="fare-range" style="opacity:0.5; font-size:0.72rem; margin-top:0.2rem">{range_note}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Input summary"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Route:** {source} → {destination}")
            st.markdown(f"**Airline:** {airline}")
            st.markdown(f"**Cabin class:** {cabin_class}")
            st.markdown(f"**Aircraft:** {aircraft}")
        with c2:
            st.markdown(f"**Stopovers:** {stop_label}")
            st.markdown(f"**Season:** {season}")
            st.markdown(f"**Booking channel:** {booking_src}")
            st.markdown(f"**Duration:** {duration} hrs  |  **Days before:** {days_before}")
        st.markdown(f"**Departure:** {dep_dt.strftime('%A, %d %B %Y')}")


# ── About ─────────────────────────────────────────────────────────────────────

with st.expander("About this app"):
    st.markdown(f"""
    Predicts total flight fares (BDT) for Bangladesh routes using a **{MODEL_LABEL}** model
    trained on 57,000+ historical flight records.

    **Features used:** Airline, Route (Origin → Destination), Cabin Class, Aircraft Type,
    Booking Channel, Travel Season, Stopovers, Flight Duration, Days Before Departure,
    Departure Month, Weekday.

    **Target:** `Total Fare (BDT)` — log₁p-transformed during training to reduce skewness;
    predictions are inverse-transformed (expm1) back to BDT for display.

    **Training pipeline:** Data validation → Imputation → Feature engineering →
    7-model comparison (GridSearchCV + RandomizedSearchCV) → Best model selected by CV R².

    Built as part of the Machine Learning Module — Amalitech Training, 2026.
    """)
