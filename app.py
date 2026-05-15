import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Prediction — Prediksi Harga Mobil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #f5f2eb;
    color: #1a1a1a;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0; max-width: 100%; }

/* ── TOPBAR ── */
.topbar {
    background: #1a1a1a;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #c8a45a;
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

.topbar-brand {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.9rem;
    color: #f5f2eb;
    letter-spacing: 2px;
    white-space: nowrap;
}

.topbar-brand span { color: #c8a45a; }

.topbar-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    letter-spacing: 1px;
    text-transform: uppercase;
    line-height: 1.4;
}

.topbar-stats {
    display: flex;
    align-items: center;
    gap: 20px;
}

.topbar-stat {
    text-align: center;
}

.topbar-stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 1px;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.topbar-stat-val {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.1rem;
    color: #f5f2eb;
    letter-spacing: 1px;
    line-height: 1;
}

.topbar-divider {
    width: 1px;
    height: 28px;
    background: #333;
}

/* ── MAIN BODY ── */
.main-body {
    padding: 32px 40px;
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 28px;
}

/* ── PANEL ── */
.panel {
    background: #fff;
    border: 1px solid #ddd8cc;
    border-radius: 2px;
    height: 100%;
    box-sizing: border-box;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    border-bottom: 1px solid #ddd8cc;
}

.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
}

.panel-number {
    font-family: 'Bebas Neue', cursive;
    font-size: 1rem;
    color: #ddd8cc;
}

.panel-body { padding: 22px; }

/* ── INPUT LABELS ── */
.stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #888 !important;
}

.stNumberInput input {
    background: #f9f7f3 !important;
    border: 1px solid #ddd8cc !important;
    border-radius: 2px !important;
    color: #1a1a1a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

.stNumberInput input:focus {
    border-color: #c8a45a !important;
    background: #fff !important;
    box-shadow: 0 0 0 2px rgba(200,164,90,0.12) !important;
}

/* ── CORR BAR ── */
.corr-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 3px;
    margin-bottom: 10px;
}

.corr-track {
    flex: 1;
    height: 2px;
    background: #ede9e0;
    position: relative;
}

.corr-fill-pos { height: 2px; background: #c8a45a; }
.corr-fill-neg { height: 2px; background: #a0522d; }

.corr-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #bbb;
    white-space: nowrap;
    min-width: 50px;
    text-align: right;
}

/* ── BUTTONS ── */
.stButton button {
    width: 100%;
    background: #1a1a1a !important;
    color: #f5f2eb !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    transition: background 0.15s !important;
}

.stButton button:hover {
    background: #c8a45a !important;
    color: #1a1a1a !important;
}

/* ── RESULT PANEL ── */
.result-hero {
    padding: 28px 22px 20px;
    border-bottom: 1px solid #ddd8cc;
}

.result-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}

.result-usd {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.8rem;
    line-height: 1;
    color: #1a1a1a;
    letter-spacing: 1px;
}

.result-idr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #c8a45a;
    margin-top: 4px;
}

/* Segment bar */
.seg-bar {
    margin: 18px 22px;
    border: 1px solid #ddd8cc;
    padding: 14px 16px;
}

.seg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}

.seg-track {
    position: relative;
    height: 4px;
    background: linear-gradient(90deg, #e8d5a3, #c8a45a, #8B4513);
    border-radius: 0;
    margin-bottom: 8px;
}

.seg-indicator {
    position: absolute;
    top: -3px;
    width: 10px;
    height: 10px;
    background: #1a1a1a;
    transform: translateX(-50%);
}

.seg-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #bbb;
}

/* Spec list */
.spec-list {
    padding: 0 22px 18px;
}

.spec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    padding: 14px 0 10px;
    border-top: 1px solid #ddd8cc;
    margin-top: 6px;
}

.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.spec-item {
    padding: 10px 12px;
    background: #f9f7f3;
    border: 1px solid #ede9e0;
}

.spec-k {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 2px;
}

.spec-v {
    font-size: 0.88rem;
    font-weight: 600;
    color: #1a1a1a;
}

/* ── IDENTITY ── */
.identity {
    margin: 0 22px 0;
    background: #1a1a1a;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}

.identity::before {
    content: 'DEVELOPER';
    position: absolute;
    right: -8px;
    top: 50%;
    transform: translateY(-50%) rotate(90deg);
    font-family: 'Bebas Neue', cursive;
    font-size: 3.5rem;
    color: rgba(255,255,255,0.04);
    letter-spacing: 4px;
    white-space: nowrap;
    pointer-events: none;
}

.identity-top {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
}

.identity-monogram {
    width: 44px;
    height: 44px;
    border: 1px solid #c8a45a;
    color: #c8a45a;
    font-family: 'Bebas Neue', cursive;
    font-size: 1.15rem;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.identity-name {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.3rem;
    letter-spacing: 1.5px;
    color: #f5f2eb;
    line-height: 1;
}

.identity-npm {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #555;
    margin-top: 4px;
    letter-spacing: 1px;
}

.identity-divider {
    height: 1px;
    background: linear-gradient(90deg, #c8a45a, transparent);
    margin-bottom: 12px;
}

.identity-meta {
    display: flex;
    gap: 20px;
}

.identity-meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.identity-meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #444;
}

.identity-meta-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #c8a45a;
}

/* ── MODEL INFO ── */
.model-info {
    margin: 0 22px 22px;
    background: #f9f7f3;
    border: 1px solid #ddd8cc;
    border-top: none;
    padding: 14px 18px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
}

.mstat {
    padding: 10px 12px;
    border-right: 1px solid #ddd8cc;
    text-align: center;
}

.mstat:last-child { border-right: none; }

.mstat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 4px;
}

.mstat-val {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.25rem;
    color: #1a1a1a;
    line-height: 1;
}

.mstat-val.gold { color: #c8a45a; }

.mstat-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    color: #bbb;
    margin-top: 2px;
}

/* ── PLACEHOLDER ── */
.placeholder {
    padding: 40px 22px;
    text-align: center;
    color: #aaa;
}

.placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    opacity: 0.3;
}

.placeholder-text {
    font-size: 0.8rem;
    line-height: 1.7;
    color: #bbb;
}

div[data-testid="column"] { padding: 0 6px; }
.element-container { margin-bottom: 0 !important; }

/* ── RESPONSIVE MOBILE ── */
@media (max-width: 768px) {
    .block-container { padding: 0 !important; }

    /* Topbar: stack brand atas, stats bawah sebagai 2x2 grid */
    .topbar {
        flex-direction: column;
        align-items: flex-start;
        gap: 14px;
        padding: 12px 16px;
    }

    .topbar-left {
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
    }

    .topbar-tag { display: none; }

    .topbar-stats {
        display: grid !important;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        width: 100%;
        border: 1px solid #2a2a2a;
    }

    .topbar-divider { display: none; }

    .topbar-stat {
        padding: 10px 14px;
        border-right: 1px solid #2a2a2a;
        border-bottom: 1px solid #2a2a2a;
        text-align: left;
    }

    .topbar-stat:nth-child(4),
    .topbar-stat:nth-child(6) { border-right: none; }

    .topbar-stat:nth-child(5),
    .topbar-stat:nth-child(6) { border-bottom: none; }

    .topbar-stat-val { font-size: 1.3rem; }

    /* Panels full width */
    .panel { margin: 0 0 12px 0; }
    .panel-body { padding: 12px; }

    div[data-testid="column"] { padding: 0 2px; }

    .result-usd { font-size: 2.6rem !important; }
    .result-idr { font-size: 0.65rem !important; }

    .spec-grid { grid-template-columns: 1fr 1fr !important; gap: 6px !important; }
    .spec-list { padding: 0 12px 12px !important; }
    .seg-bar { margin: 12px !important; }

    .identity { margin: 0 !important; }
    .identity-meta { flex-wrap: wrap; gap: 12px; }

    .model-info { margin: 0 !important; }

    .stButton button { font-size: 0.72rem !important; padding: 0.6rem !important; }
    .topbar-brand { font-size: 1.5rem !important; }
}

@media (max-width: 480px) {
    .result-usd { font-size: 2rem !important; }
    .spec-grid { grid-template-columns: 1fr 1fr !important; }
    .identity-meta { gap: 8px; }
    .mstat-val { font-size: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Load / Train Model ────────────────────────────────────────────────────────
FEATURE_COLS = ['Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_efficiency']
FEATURE_META = {
    'Engine_size'    : dict(label='Engine Size',     unit='Liter', min=1.0,  max=8.0,  step=0.1,  default=2.98),
    'Horsepower'     : dict(label='Horsepower',      unit='HP',    min=50,   max=500,  step=1.0,  default=156.5),
    'Wheelbase'      : dict(label='Wheelbase',       unit='inch',  min=90.0, max=140., step=0.1,  default=114.64),
    'Width'          : dict(label='Width',           unit='inch',  min=60.0, max=90.0, step=0.1,  default=72.22),
    'Length'         : dict(label='Length',          unit='inch',  min=140., max=230., step=0.1,  default=195.12),
    'Curb_weight'    : dict(label='Curb Weight',     unit='K lbs', min=1.0,  max=6.0,  step=0.01, default=3.34),
    'Fuel_efficiency': dict(label='Fuel Efficiency', unit='mpg',   min=10.0, max=60.0, step=0.1,  default=24.1),
}
KORRELASI = {'Engine_size':0.626,'Horsepower':0.837,'Wheelbase':0.489,'Width':0.441,'Length':0.398,'Curb_weight':0.523,'Fuel_efficiency':-0.492}

@st.cache_resource
def load_model():
    pkl_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if os.path.exists(pkl_path):
        try:
            d = joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
        csv_path = os.path.join(os.path.dirname(__file__), 'Car_sales.xls')
        df = pd.read_csv(csv_path)
        df.dropna(how='all', inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        feature_cols = d.get('feature_cols', FEATURE_COLS)
        X = df[feature_cols]
        y = df['Price_in_thousands']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = d['model'].predict(X_test)
        stats = {
            'r2'     : round(float(r2_score(y_test, y_pred)), 4),
            'rmse'   : round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            'n_train': len(X_train),
            'n_test' : len(X_test),
        }
        return d['model'], stats

    csv_path = os.path.join(os.path.dirname(__file__), 'Car_sales.xls')
    df = pd.read_csv(csv_path)
    df.dropna(how='all', inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    X = df[FEATURE_COLS]
    y = df['Price_in_thousands']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    stats = {
        'r2'     : round(float(r2_score(y_test, y_pred)), 4),
        'rmse'   : round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        'n_train': len(X_train),
        'n_test' : len(X_test),
    }
    joblib.dump({'model': model, 'feature_cols': FEATURE_COLS}, 'model.pkl')
    return model, stats

model, stats = load_model()
r2   = stats.get('r2',   0.734)
rmse = stats.get('rmse', 9.561)
n_tr = stats.get('n_train', 125)

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-brand">CAR PRICE<span> PREDICTION</span></div>
        <div class="topbar-tag">Sistem Prediksi Harga Mobil · Final Project Sains Data</div>
    </div>
    <div class="topbar-stats">
        <div class="topbar-stat">
            <div class="topbar-stat-label">Algoritma</div>
            <div class="topbar-stat-val">Lin.Reg</div>
        </div>
        <div class="topbar-divider"></div>
        <div class="topbar-stat">
            <div class="topbar-stat-label">R² Score</div>
            <div class="topbar-stat-val" style="color:#c8a45a;">{r2}</div>
        </div>
        <div class="topbar-divider"></div>
        <div class="topbar-stat">
            <div class="topbar-stat-label">RMSE</div>
            <div class="topbar-stat-val">${rmse}K</div>
        </div>
        <div class="topbar-divider"></div>
        <div class="topbar-stat">
            <div class="topbar-stat-label">Data Latih</div>
            <div class="topbar-stat-val">{n_tr} baris</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.5, 1], gap="medium")

with col_left:
    st.markdown("""
    <div style="margin:0 10px">
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">⚙ Spesifikasi Kendaraan</span>
            <span class="panel-number">01</span>
        </div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    inputs = {}
    c1, c2 = st.columns(2)
    cols_ui = [c1, c2]
    for i, (key, meta) in enumerate(FEATURE_META.items()):
        with cols_ui[i % 2]:
            val = st.number_input(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(meta['min']),
                max_value=float(meta['max']),
                value=float(meta['default']),
                step=float(meta['step']),
                key=key,
                format="%.2f" if meta['step'] < 1 else "%.1f"
            )
            corr = KORRELASI[key]
            pct  = min(abs(corr) * 100, 100)
            clr  = "corr-fill-pos" if corr >= 0 else "corr-fill-neg"
            st.markdown(f"""
            <div class="corr-row">
                <div class="corr-track">
                    <div class="{clr}" style="width:{pct}%"></div>
                </div>
                <span class="corr-num">r = {corr:+.3f}</span>
            </div>""", unsafe_allow_html=True)
            inputs[key] = val

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        hitung = st.button("Hitung Harga Mobil", use_container_width=True)
    with btn_c2:
        reset = st.button("Reset ke Default", use_container_width=True)

    if reset:
        for key, meta in FEATURE_META.items():
            st.session_state[key] = float(meta['default'])
        st.rerun()

with col_right:
    # Hitung prediksi dulu di luar HTML
    if hitung:
        X_input  = pd.DataFrame([inputs])
        pred_k   = float(model.predict(X_input)[0])
        pred_usd = pred_k * 1000
        pred_idr = pred_usd * 17_603

        if pred_usd < 15000:
            seg_label, seg_pct = "Budget", 12
        elif pred_usd < 35000:
            seg_label, seg_pct = "Menengah", 48
        else:
            seg_label, seg_pct = "Premium", 85

        # Build specs rows
        specs_rows = ""
        for key, meta in FEATURE_META.items():
            val_fmt = f"{inputs[key]:.2f}"
            specs_rows += (
                '<div class="spec-item">'
                '<div class="spec-k">' + meta['label'] + '</div>'
                '<div class="spec-v">' + val_fmt + ' <span style="font-size:0.65rem;color:#aaa;font-weight:400">' + meta['unit'] + '</span></div>'
                '</div>'
            )

        # Build full right panel HTML (no nested f-strings)
        usd_str = "${:,.0f}".format(pred_usd)
        idr_str = "Rp {:,.0f}".format(pred_idr)
        r2_str  = "{:.1f}%".format(r2 * 100)
        rmse_str = "${:.3f}K".format(rmse)

        # shared identity + model stats block
        identity_block = (
            '<div class="identity">'
              '<div class="identity-top">'
                '<div class="identity-monogram">RNA</div>'
                '<div>'
                  '<div class="identity-name">Rossi Nur Ajizah</div>'
                  '<div class="identity-npm">NPM · 237006003</div>'
                '</div>'
              '</div>'
              '<div class="identity-divider"></div>'
              '<div class="identity-meta">'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Matakuliah</span>'
                  '<span class="identity-meta-val">Sains Data</span>'
                '</div>'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Metode</span>'
                  '<span class="identity-meta-val">Lin. Regression</span>'
                '</div>'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Status</span>'
                  '<span class="identity-meta-val">Final Project</span>'
                '</div>'
              '</div>'
            '</div>'
            '<div class="model-info">'
              '<div class="mstat">'
                '<div class="mstat-label">R² Score</div>'
                '<div class="mstat-val gold">' + r2_str + '</div>'
                '<div class="mstat-sub">akurasi model</div>'
              '</div>'
              '<div class="mstat">'
                '<div class="mstat-label">RMSE</div>'
                '<div class="mstat-val">' + rmse_str + '</div>'
                '<div class="mstat-sub">rata-rata error</div>'
              '</div>'
              '<div class="mstat">'
                '<div class="mstat-label">Top Feature</div>'
                '<div class="mstat-val" style="font-size:0.85rem;font-family:\'Space Grotesk\',sans-serif;font-weight:700;">HP</div>'
                '<div class="mstat-sub">r = 0.837</div>'
              '</div>'
            '</div>'
        )

        html = (
            '<div class="panel">'
              '<div class="panel-header">'
                '<span class="panel-title">$ Perkiraan Harga</span>'
                '<span class="panel-number">02</span>'
              '</div>'
              '<div class="result-hero">'
                '<div class="result-eyebrow">Estimasi Harga Mobil</div>'
                '<div class="result-usd">' + usd_str + '</div>'
                '<div class="result-idr">≈ ' + idr_str + '</div>'
              '</div>'
              '<div class="seg-bar">'
                '<div class="seg-label">Segmen Harga — ' + seg_label + '</div>'
                '<div class="seg-track">'
                  '<div class="seg-indicator" style="left:' + str(seg_pct) + '%"></div>'
                '</div>'
                '<div class="seg-labels">'
                  '<span>Budget</span><span>Menengah</span><span>Premium</span>'
                '</div>'
              '</div>'
              '<div class="spec-list">'
                '<div class="spec-title">Ringkasan Spesifikasi</div>'
                '<div class="spec-grid">' + specs_rows + '</div>'
              '</div>'
              + identity_block +
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    else:
        r2_str  = "{:.1f}%".format(r2 * 100)
        rmse_str = "${:.3f}K".format(rmse)

        identity_block = (
            '<div class="identity">'
              '<div class="identity-top">'
                '<div class="identity-monogram">RNA</div>'
                '<div>'
                  '<div class="identity-name">Rossi Nur Ajizah</div>'
                  '<div class="identity-npm">NPM · 237006003</div>'
                '</div>'
              '</div>'
              '<div class="identity-divider"></div>'
              '<div class="identity-meta">'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Matakuliah</span>'
                  '<span class="identity-meta-val">Sains Data</span>'
                '</div>'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Metode</span>'
                  '<span class="identity-meta-val">Lin. Regression</span>'
                '</div>'
                '<div class="identity-meta-item">'
                  '<span class="identity-meta-label">Status</span>'
                  '<span class="identity-meta-val">Final Project</span>'
                '</div>'
              '</div>'
            '</div>'
            '<div class="model-info">'
              '<div class="mstat">'
                '<div class="mstat-label">R² Score</div>'
                '<div class="mstat-val gold">' + r2_str + '</div>'
                '<div class="mstat-sub">akurasi model</div>'
              '</div>'
              '<div class="mstat">'
                '<div class="mstat-label">RMSE</div>'
                '<div class="mstat-val">' + rmse_str + '</div>'
                '<div class="mstat-sub">rata-rata error</div>'
              '</div>'
              '<div class="mstat">'
                '<div class="mstat-label">Top Feature</div>'
                '<div class="mstat-val" style="font-size:0.85rem;font-family:\'Space Grotesk\',sans-serif;font-weight:700;">HP</div>'
                '<div class="mstat-sub">r = 0.837</div>'
              '</div>'
            '</div>'
        )

        html = (
            '<div class="panel">'
              '<div class="panel-header">'
                '<span class="panel-title">$ Perkiraan Harga</span>'
                '<span class="panel-number">02</span>'
              '</div>'
              '<div class="placeholder">'
                '<div class="placeholder-icon">◈</div>'
                '<div class="placeholder-text">'
                  'Masukkan spesifikasi kendaraan<br>'
                  'lalu klik <strong>Hitung Harga Mobil</strong>'
                '</div>'
              '</div>'
              + identity_block +
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
