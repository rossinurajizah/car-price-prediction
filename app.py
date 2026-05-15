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
    page_title="AutoPrice — Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0f1a;
    color: #e8eaf0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #131929 0%, #1a2236 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(245,166,35,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 40%, #f5a623);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}

.hero-sub {
    color: #6b7592;
    font-size: 0.9rem;
    margin-top: 6px;
}

/* Stat pills */
.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}

.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-green { background: #27d98b; box-shadow: 0 0 8px #27d98b; }
.dot-blue  { background: #4a9eff; box-shadow: 0 0 8px #4a9eff; }
.dot-amber { background: #f5a623; box-shadow: 0 0 8px #f5a623; }

/* Cards */
.card {
    background: #131929;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: #e8eaf0;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #131929 0%, #1a2236 100%);
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-box::before {
    content: '';
    position: absolute;
    top: -30px; left: -30px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(245,166,35,0.06) 0%, transparent 70%);
}

.result-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7592;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f5a623, #e8452c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.result-idr {
    color: #6b7592;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

.divider {
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 1.2rem 0;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
}

.spec-key { color: #6b7592; font-size: 0.8rem; }
.spec-val { color: #e8eaf0; font-size: 0.85rem; font-weight: 500; }

/* Identity card */
.identity-card {
    background: linear-gradient(135deg, rgba(74,158,255,0.08), rgba(39,217,139,0.06));
    border: 1px solid rgba(74,158,255,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}

.id-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4a9eff;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.id-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: white;
}

.id-npm { color: #6b7592; font-size: 0.85rem; margin-top: 3px; }

/* Corr bar */
.corr-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
}
.corr-bar-bg {
    flex: 1;
    height: 3px;
    background: #1a2236;
    border-radius: 3px;
    overflow: hidden;
}
.corr-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4a9eff, #27d98b);
    border-radius: 3px;
}
.corr-val { font-size: 0.7rem; color: #6b7592; white-space: nowrap; }

/* Streamlit input overrides */
.stNumberInput label { color: #6b7592 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.6px; }
.stNumberInput input {
    background: #1a2236 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
.stNumberInput input:focus { border-color: #f5a623 !important; }

.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #f5a623, #e8452c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}

.stButton button:hover { opacity: 0.9; transform: translateY(-1px); }

div[data-testid="column"] { padding: 0 0.5rem; }
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
        # Support joblib (dari Colab) maupun pickle
        try:
            d = joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
        # Hitung stats dari model yang sudah ada
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

    # Kalau model.pkl tidak ada → train langsung dari CSV
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

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:2.8rem;">🚗</div>
        <div>
            <div class="hero-title">AutoPrice Predictor</div>
            <div class="hero-sub">Sistem Prediksi Harga Mobil — Final Project Matakuliah Sains Data</div>
        </div>
    </div>
    <div class="stats-row">
        <div class="stat-pill"><span class="dot dot-green"></span> Linear Regression</div>
        <div class="stat-pill"><span class="dot dot-blue"></span> R² Score <strong style="color:#fff;margin-left:4px">{r2}</strong></div>
        <div class="stat-pill"><span class="dot dot-amber"></span> RMSE <strong style="color:#fff;margin-left:4px">${rmse}K</strong></div>
        <div class="stat-pill"><span class="dot dot-green"></span> Data Training <strong style="color:#fff;margin-left:4px">{n_tr} baris</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.4, 1], gap="medium")

with col_left:
    st.markdown('<div class="card"><div class="card-title">⚙️ Spesifikasi Kendaraan</div>', unsafe_allow_html=True)

    inputs = {}
    c1, c2 = st.columns(2)
    cols = [c1, c2]
    for i, (key, meta) in enumerate(FEATURE_META.items()):
        with cols[i % 2]:
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
            <div class="corr-wrap">
                <div class="corr-bar-bg"><div class="corr-bar-fill" style="width:{pct}%"></div></div>
                <span class="corr-val">r={KORRELASI[key]}</span>
            </div>""", unsafe_allow_html=True)
            inputs[key] = val

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        hitung = st.button("🔍 Hitung Harga Mobil", use_container_width=True)
    with btn_col2:
        reset = st.button("↺ Reset ke Rekomendasi", use_container_width=True)

    if reset:
        for key, meta in FEATURE_META.items():
            st.session_state[key] = float(meta['default'])
        st.rerun()

with col_right:
    # Result
    if hitung:
        X_input = pd.DataFrame([inputs])
        pred_k   = float(model.predict(X_input)[0])
        pred_usd = pred_k * 1000
        pred_idr = pred_usd * 17_603

        specs_html = ""
        for key, meta in FEATURE_META.items():
            specs_html += f"""
            <div class="spec-row">
                <span class="spec-key">{meta['label']}</span>
                <span class="spec-val">{inputs[key]:.2f} {meta['unit']}</span>
            </div>"""

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Perkiraan Harga Mobil</div>
            <div class="result-price">${pred_usd:,.0f}</div>
            <div class="result-idr">≈ Rp {pred_idr:,.0f}</div>
            <div class="divider"></div>
            {specs_html}
            <div class="divider"></div>
            <div style="font-size:0.72rem;color:#6b7592;">* Kurs USD/IDR: Rp 17.603</div>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"✅ Prediksi berhasil! Harga: ${pred_usd:,.0f}")

    else:
        st.markdown("""
        <div class="result-box" style="padding: 3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:12px;">💡</div>
            <div style="color:#6b7592; font-size:0.9rem;">
                Masukkan spesifikasi mobil<br>lalu klik <strong style="color:#f5a623">Hitung Harga Mobil</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Identity card
    st.markdown("""
    <div class="identity-card">
        <div class="id-label"> Sistem Ini Dibuat Oleh</div>
        <div class="id-name">Rossi Nur Ajizah</div>
        <div class="id-npm">NPM : 237006003</div>
    </div>

    <div class="card" style="margin-top:1rem; padding: 1.2rem 1.5rem;">
        <div class="card-title" style="font-size:0.85rem; margin-bottom:0.8rem">📊 Tentang Model</div>
        <div style="font-size:0.8rem; color:#6b7592; line-height:1.7">
            Model <strong style="color:#e8eaf0">Linear Regression</strong> dilatih menggunakan
            dataset <em>Car_sales.xls</em>.<br><br>
            Variabel paling berpengaruh:
            <strong style="color:#f5a623">Horsepower (r=0.837)</strong>.<br><br>
            <span style="color:#27d98b">R² = {r2_pct:.1f}%</span> variasi harga dijelaskan model ini.
        </div>
    </div>
    """.format(r2_pct=r2*100), unsafe_allow_html=True)
