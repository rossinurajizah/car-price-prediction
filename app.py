import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1100px; margin: auto; }

/* ── Header ── */
.top-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.5rem;
}
.top-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
}
.top-sub {
    font-size: 0.82rem;
    color: #7d8590;
    margin-top: 4px;
}
.live-badge {
    display: inline-block;
    background: #1f6feb;
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 999px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Panel ── */
.panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem;
}
.panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Input row ── */
.inp-row {
    margin-bottom: 1.1rem;
}
.inp-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.inp-name { font-size: 0.82rem; color: #c9d1d9; }
.inp-val {
    font-size: 0.75rem;
    font-weight: 600;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 999px;
    padding: 2px 10px;
    color: #f0b429;
}
.corr-bar-bg {
    height: 3px;
    background: #21262d;
    border-radius: 3px;
    margin-top: 4px;
    overflow: hidden;
}
.corr-bar-fill {
    height: 100%;
    border-radius: 3px;
}

/* ── Price box ── */
.price-box {
    background: linear-gradient(135deg, #1c2333, #1f2d3d);
    border: 1px solid #1f6feb44;
    border-radius: 10px;
    text-align: center;
    padding: 1.4rem 1rem;
    margin-bottom: 1rem;
}
.price-label {
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 6px;
}
.price-usd {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0b429;
    line-height: 1;
}
.price-idr {
    font-size: 0.8rem;
    color: #7d8590;
    margin-top: 5px;
}
.price-segment {
    display: inline-block;
    margin-top: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 999px;
}

/* ── Segment bar ── */
.seg-wrap { margin: 0.8rem 0; }
.seg-title { font-size: 0.7rem; color: #7d8590; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.8px; }
.seg-bar {
    height: 6px;
    background: linear-gradient(90deg, #238636, #f0b429, #1f6feb);
    border-radius: 3px;
    position: relative;
    margin-bottom: 4px;
}
.seg-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #7d8590;
}

/* ── Spec grid ── */
.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 1rem;
}
.spec-cell {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 10px;
}
.spec-cell-label { font-size: 0.68rem; color: #7d8590; margin-bottom: 3px; }
.spec-cell-val { font-size: 0.88rem; font-weight: 600; color: #e6edf3; }

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 1rem;
}
.metric-cell {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}
.metric-val { font-size: 1.2rem; font-weight: 700; color: #58a6ff; }
.metric-label { font-size: 0.68rem; color: #7d8590; margin-top: 2px; }

/* ── Identity ── */
.identity-box {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.identity-title { font-size: 0.65rem; color: #7d8590; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.identity-name { font-size: 0.9rem; font-weight: 600; color: #e6edf3; }
.identity-npm { font-size: 0.78rem; color: #7d8590; margin-top: 2px; }

/* ── Button ── */
.stButton button {
    width: 100%;
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.65rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    margin-top: 0.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* ── Number input overrides ── */
.stNumberInput label {
    color: #c9d1d9 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.stNumberInput input {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.9rem !important;
}
.stNumberInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

div[data-testid="column"] { padding: 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_COLS = ['Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_efficiency']
FEATURE_META = {
    'Engine_size'    : dict(label='Engine Size',     unit='L',     min=1.0,  max=8.0,  step=0.1,  default=2.98,   color='#f0b429'),
    'Horsepower'     : dict(label='Horsepower',      unit='HP',    min=50,   max=500,  step=1.0,  default=156.5,  color='#f0b429'),
    'Wheelbase'      : dict(label='Wheelbase',       unit='in',    min=90.0, max=140., step=0.1,  default=114.64, color='#58a6ff'),
    'Width'          : dict(label='Width',           unit='in',    min=60.0, max=90.0, step=0.1,  default=72.22,  color='#58a6ff'),
    'Length'         : dict(label='Length',          unit='in',    min=140., max=230., step=0.1,  default=195.12, color='#58a6ff'),
    'Curb_weight'    : dict(label='Curb Weight',     unit='K lbs', min=1.0,  max=6.0,  step=0.01, default=3.34,   color='#3fb950'),
    'Fuel_efficiency': dict(label='Fuel Efficiency', unit='mpg',   min=10.0, max=60.0, step=0.1,  default=24.1,   color='#da3633'),
}
KORRELASI = {'Engine_size':0.626,'Horsepower':0.837,'Wheelbase':0.489,'Width':0.441,'Length':0.398,'Curb_weight':0.523,'Fuel_efficiency':-0.492}

@st.cache_resource
def load_model():
    pkl_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    csv_path = os.path.join(os.path.dirname(__file__), 'Car_sales.xls')

    def get_stats(model, csv_path):
        df = pd.read_csv(csv_path)
        df.dropna(how='all', inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        X = df[FEATURE_COLS]
        y = df['Price_in_thousands']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        return {
            'r2'     : round(float(r2_score(y_test, y_pred)), 4),
            'rmse'   : round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            'n_train': len(X_train),
        }

    if os.path.exists(pkl_path):
        try:
            d = joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
        stats = get_stats(d['model'], csv_path)
        return d['model'], stats

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
    }
    joblib.dump({'model': model, 'feature_cols': FEATURE_COLS}, pkl_path)
    return model, stats

model, stats = load_model()
r2   = stats.get('r2',   0.734)
rmse = stats.get('rmse', 9.561)
n_tr = stats.get('n_train', 125)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div class="top-title">
        Prediksi Harga Mobil
        <span class="live-badge">LIVE</span>
    </div>
    <div class="top-sub">Matakuliah Sains Data — Final Project</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.1], gap="medium")

with col_left:
    st.markdown('<div class="panel"><div class="panel-title">⚙ Spesifikasi Mobil</div>', unsafe_allow_html=True)

    inputs = {}
    for key, meta in FEATURE_META.items():
        val = st.number_input(
            f"{meta['label']} ({meta['unit']})",
            min_value=float(meta['min']),
            max_value=float(meta['max']),
            value=float(meta['default']),
            step=float(meta['step']),
            key=key,
            format="%.2f" if meta['step'] < 1 else "%.1f"
        )
        pct = min(abs(KORRELASI[key]) * 100, 100)
        st.markdown(f"""
        <div class="corr-bar-bg">
            <div class="corr-bar-fill" style="width:{pct}%;background:{meta['color']};"></div>
        </div>""", unsafe_allow_html=True)
        inputs[key] = val

    st.markdown('</div>', unsafe_allow_html=True)
    hitung = st.button("▶  Hitung Harga Mobil", use_container_width=True)

with col_right:
    if hitung:
        X_input = pd.DataFrame([inputs])
        pred_k   = float(model.predict(X_input)[0])
        pred_usd = pred_k * 1000
        pred_idr = pred_usd * 17_603

        # Segment
        if pred_usd < 15000:
            seg_label, seg_color, seg_bg, seg_pos = "Budget", "#3fb950", "rgba(63,185,80,0.15)", "10%"
        elif pred_usd < 35000:
            seg_label, seg_color, seg_bg, seg_pos = "Menengah", "#f0b429", "rgba(240,180,41,0.15)", "50%"
        else:
            seg_label, seg_color, seg_bg, seg_pos = "Premium", "#58a6ff", "rgba(88,166,255,0.15)", "90%"

        specs_cells = ""
        for key, meta in FEATURE_META.items():
            specs_cells += f"""
            <div class="spec-cell">
                <div class="spec-cell-label">{meta['label']}</div>
                <div class="spec-cell-val">{inputs[key]:.2f} {meta['unit']}</div>
            </div>"""

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">$ Perkiraan Harga</div>

            <div class="price-box">
                <div class="price-label">Estimasi Harga Mobil</div>
                <div class="price-usd">${pred_usd:,.0f}</div>
                <div class="price-idr">≈ Rp {pred_idr:,.0f}</div>
                <span class="price-segment" style="background:{seg_bg};color:{seg_color};">
                    ● Segmen {seg_label}
                </span>
            </div>

            <div class="seg-wrap">
                <div class="seg-title">Segmen Harga</div>
                <div class="seg-bar">
                    <div style="position:absolute;top:-3px;left:{seg_pos};width:12px;height:12px;background:{seg_color};border-radius:50%;border:2px solid #0d1117;transform:translateX(-50%);"></div>
                </div>
                <div class="seg-labels"><span>Budget</span><span>Menengah</span><span>Premium</span></div>
            </div>

            <div class="panel-title" style="margin-top:1rem;">Ringkasan Spesifikasi</div>
            <div class="spec-grid">{specs_cells}</div>

            <div class="panel-title">Performa Model</div>
            <div class="metric-grid">
                <div class="metric-cell">
                    <div class="metric-val">{r2:.4f}</div>
                    <div class="metric-label">R² Score<br>{r2*100:.1f}% akurasi</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-val">${rmse:.2f}K</div>
                    <div class="metric-label">RMSE<br>rata-rata error</div>
                </div>
            </div>

            <div class="identity-box">
                <div class="identity-title">Sistem ini dibuat oleh</div>
                <div class="identity-name">Rossi Nur Ajizah</div>
                <div class="identity-npm">NPM · 237006003</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">$ Perkiraan Harga</div>

            <div class="price-box" style="padding:2.5rem 1rem;">
                <div style="font-size:2rem;margin-bottom:8px;">—</div>
                <div class="price-label">Masukkan spesifikasi lalu klik<br>Hitung Harga Mobil</div>
            </div>

            <div class="panel-title" style="margin-top:1.2rem;">Performa Model</div>
            <div class="metric-grid">
                <div class="metric-cell">
                    <div class="metric-val">{r2:.4f}</div>
                    <div class="metric-label">R² Score<br>{r2*100:.1f}% akurasi</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-val">${rmse:.2f}K</div>
                    <div class="metric-label">RMSE<br>rata-rata error</div>
                </div>
            </div>

            <div class="identity-box">
                <div class="identity-title">Sistem ini dibuat oleh</div>
                <div class="identity-name">Rossi Nur Ajizah</div>
                <div class="identity-npm">NPM · 237006003</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
