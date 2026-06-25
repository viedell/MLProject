import os
import base64
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Mengabaikan peringatan agar UI tetap bersih
warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFIGURASI HALAMAN & MANAJEMEN TEMA (MINIMALIST SAAS STYLE)
# =============================================================================
st.set_page_config(
    page_title="Enterprise Customer Insight AI",
    page_icon="https://img.icons8.com/color/48/artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """
    Fungsi untuk menyuntikkan custom CSS ke dalam Streamlit.
    Mengatur font, warna latar, styling card, dan komponen visual lainnya
    agar menyerupai aplikasi SaaS modern minimalis (tanpa emoji).
    """
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC;
        }
        
        /* Modern Header Banner */
        .hero-banner {
            background-color: #0F172A;
            border-bottom: 2px solid #E2E8F0;
            padding: 35px 30px;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
            color: #F8FAFC;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            font-weight: 400;
            opacity: 0.85;
            max-width: 900px;
            color: #E2E8F0;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            transition: all 0.2s ease-in-out;
        }
        .metric-container:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        }
        .metric-title {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.75px;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0F172A;
        }

        /* Minimalist Insight Cards */
        .insight-card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            border: 1px solid #E2E8F0;
            border-left: 5px solid #CBD5E1;
            transition: all 0.2s ease;
        }
        .insight-card:hover {
            border-color: #CBD5E1;
        }
        .insight-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 8px;
        }
        .insight-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #0F172A;
        }
        .insight-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }
        .insight-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
            background-color: #F8FAFC;
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #F1F5F9;
        }
        .stat-item {
            text-align: center;
        }
        .stat-label {
            font-size: 0.7rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .stat-val {
            font-size: 1.05rem;
            font-weight: 600;
            color: #0F172A;
            margin-top: 2px;
        }
        .strategy-box {
            background-color: #F8FAFC;
            border-left: 3px solid #6366F1;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.5;
            border: 1px solid #E2E8F0;
            border-left-width: 3px;
        }

        /* Step Card for Tutorial */
        .step-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 10px;
        }
        .step-badge {
            background-color: #0F172A;
            color: white;
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: 600;
            margin-right: 8px;
            font-size: 0.85rem;
        }

        /* Tabs Styling Override */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background-color: #F1F5F9;
            border-radius: 6px 6px 0px 0px;
            padding-top: 8px;
            padding-bottom: 8px;
            font-weight: 500;
            color: #475569;
            border: 1px solid #E2E8F0;
            border-bottom: none;
            transition: all 0.15s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            color: #4F46E5;
            font-weight: 600;
            border-top: 2px solid #4F46E5;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_custom_css()

# Render Modern Header (No Emojis)
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Enterprise Customer Insight AI</div>
    <div class="hero-subtitle">Platform analitik data terstruktur bertenaga Machine Learning untuk memetakan perilaku dan preferensi segmen pelanggan secara presisi.</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATA PIPELINE & PREPROCESSING
# =============================================================================

@st.cache_data(ttl=3600)
def load_and_clean_data(file_source):
    """Memuat dataset dari CSV dan menghapus duplikasi data."""
    try:
        df = pd.read_csv(file_source)
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

# =============================================================================
# 3. SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.markdown("### Konfigurasi Sistem")
st.sidebar.markdown("Unggah database pelanggan Anda dalam format CSV.")

DEFAULT_CSV = "Mall_Customers.csv"
uploaded_file = st.sidebar.file_uploader("Unggah Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"Data {uploaded_file.name} berhasil dimuat.")
else:
    if os.path.exists(DEFAULT_CSV):
        df_raw = load_and_clean_data(DEFAULT_CSV)
        st.sidebar.info("Menggunakan database sampel default.")
    else:
        st.sidebar.error("Dataset default tidak ditemukan. Harap unggah data.")
        st.stop()

# Deteksi Kolom Numerik
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
non_id_numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]
if len(non_id_numeric_cols) < 2:
    non_id_numeric_cols = numeric_cols

if len(non_id_numeric_cols) < 2:
    st.error("Dataset minimal harus memiliki 2 kolom bertipe angka.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Fitur Analisis")

# Multi-select untuk fitur klastering
default_selections = []
for kw in ["age", "income", "score", "belanja", "pendapatan", "usia"]:
    for col in non_id_numeric_cols:
        if kw in col.lower() and col not in default_selections:
            default_selections.append(col)

if len(default_selections) < 2:
    default_selections = non_id_numeric_cols[:min(3, len(non_id_numeric_cols))]

SELECTED_FEATURES = st.sidebar.multiselect(
    "Pilih Fitur Kunci", 
    options=non_id_numeric_cols, 
    default=default_selections
)

if len(SELECTED_FEATURES) < 2:
    st.sidebar.warning("Harap pilih minimal 2 fitur untuk segmentasi.")
    st.stop()

# Preprocessing: Pilihan Normalisasi & Outlier Filtering
st.sidebar.markdown("---")
st.sidebar.markdown("### Preprocessing Data")
scaler_choice = st.sidebar.selectbox(
    "Metode Normalisasi Skala",
    options=["StandardScaler (Z-Score)", "MinMaxScaler (0-1 Scale)"],
    index=0
)

filter_outliers = st.sidebar.checkbox("Filter Data Outlier (Z-Score)", value=False)
z_threshold = 3.0
if filter_outliers:
    z_threshold = st.sidebar.slider("Ambang Batas Outlier (Std Dev)", min_value=1.5, max_value=4.0, value=3.0, step=0.5)

# Terapkan Preprocessing & Pembersihan Data
if filter_outliers:
    # Cari nilai Z-score untuk fitur terpilih
    mean_vals = df_raw[SELECTED_FEATURES].mean()
    std_vals = df_raw[SELECTED_FEATURES].std()
    z_scores = np.abs((df_raw[SELECTED_FEATURES] - mean_vals) / std_vals)
    
    # Filter baris yang tidak melebihi ambang batas di seluruh fitur terpilih
    filtered_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_raw[filtered_mask].dropna(subset=SELECTED_FEATURES).reset_index(drop=True)
    removed_count = len(df_raw) - len(df_clean)
    st.sidebar.caption(f"Berhasil memfilter {removed_count} baris pencilan.")
else:
    df_clean = df_raw.dropna(subset=SELECTED_FEATURES).reset_index(drop=True)

X = df_clean[SELECTED_FEATURES]

# Konfigurasi Parameter K-Means
st.sidebar.markdown("---")
st.sidebar.markdown("### Parameter K-Means")
n_clusters = st.sidebar.slider("Jumlah Segmen (K)", min_value=2, max_value=10, value=5, step=1)
max_iter = st.sidebar.number_input("Iterasi Maksimal", min_value=100, max_value=1000, value=300, step=50)
random_seed = st.sidebar.number_input("State Pengacak (Seed)", value=42)

# =============================================================================
# 4. INISIALISASI SESSION STATE & PEMETAAN LABEL KLASTER CUSTOM
# =============================================================================
# Inisialisasi session state untuk nama klaster kustom jika belum ada
if 'cluster_names' not in st.session_state:
    st.session_state['cluster_names'] = {}

# Siapkan nama label default untuk meminimalkan jeda
custom_names = {}
for idx in range(n_clusters):
    key = f"cname_{idx}"
    if key not in st.session_state:
        st.session_state[key] = f"Segment {idx+1}"
    custom_names[idx] = st.session_state[key]

# =============================================================================
# 5. PEMROSESAN MODEL MACHINE LEARNING
# =============================================================================
# Scaling Fitur
scaler = StandardScaler() if "StandardScaler" in scaler_choice else MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Model K-Means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, random_state=random_seed, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Tambahkan label hasil ke DataFrame
df_clean["Cluster_ID"] = cluster_labels
df_clean["Cluster_Name"] = df_clean["Cluster_ID"].map(custom_names)

# Evaluasi Metrik
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

# Kamus Profil Persona Bisnis (Default fallback untuk dataset Mall Customers)
PERSONA_PRESETS = {
    0: {"name": "Premium / High Rollers", "color": "#F43F5E", "desc": "Pendapatan tinggi, pengeluaran tinggi. Fokus pada program apresiasi pelanggan setia, peluncuran produk premium eksklusif, dan layanan personal."},
    1: {"name": "Thrifty Savers", "color": "#3B82F6", "desc": "Pendapatan tinggi, pengeluaran rendah. Targetkan dengan promo penawaran bernilai tinggi, cashbacks, dan keunggulan utilitas jangka panjang."},
    2: {"name": "Core Loyalists", "color": "#10B981", "desc": "Pendapatan rata-rata, pengeluaran stabil. Berikan program poin loyalitas berjenjang secara berkala untuk menjaga retensi jangka panjang."},
    3: {"name": "Spontaneous Shoppers", "color": "#F59E0B", "desc": "Pendapatan rendah, pengeluaran tinggi. Tawarkan produk bundling, diskon kilat, dan opsi pembayaran fleksibel."},
    4: {"name": "Frugal Essentials", "color": "#64748B", "desc": "Pendapatan rendah, pengeluaran rendah. Fokus pada promosi produk kebutuhan dasar dengan harga hemat dan diskon volume produk pokok."}
}

def get_persona_styling(cluster_id, total_k):
    """Mengembalikan warna aksen dan rekomendasi strategi berdasarkan ID cluster secara konsisten."""
    is_mall_dataset = any("income" in f.lower() or "pendapatan" in f.lower() for f in SELECTED_FEATURES)
    
    if total_k == 5 and is_mall_dataset and cluster_id in PERSONA_PRESETS:
        return PERSONA_PRESETS[cluster_id]
        
    colors = ["#6366F1", "#3B82F6", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#64748B"]
    color = colors[cluster_id % len(colors)]
    return {
        "name": f"Segment {cluster_id+1}",
        "color": color,
        "desc": f"Segmen data klaster ke-{cluster_id+1}. Analisis profil sentroid lengkap dapat ditinjau pada tab AI Clustering Engine."
    }

# Mapping Warna Plotly
color_map = {custom_names[k]: get_persona_styling(k, n_clusters)["color"] for k in range(n_clusters)}

# =============================================================================
# 6. ANTARMUKA UTAMA (TAB SYSTEM)
# =============================================================================
tab1, tab_tutorial, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary", 
    "Data Tutorial & Schema",
    "Exploratory Data Analysis", 
    "AI Clustering Engine", 
    "Predictive Simulator", 
    "Data Export & Reports"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### Analisis Strategis Segmen Pelanggan")
    st.markdown("Berikut adalah visualisasi pangsa pasar dan karakteristik utama kelompok pelanggan yang berhasil diidentifikasi.")
    
    # Ringkasan KPI
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Total Pelanggan Teranalisis</div><div class="metric-value">{len(df_clean)}</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Jumlah Segmen Model</div><div class="metric-value">{n_clusters}</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        target_age_col = next((f for f in SELECTED_FEATURES if "age" in f.lower() or "usia" in f.lower()), SELECTED_FEATURES[0])
        avg_val1 = round(df_clean[target_age_col].mean(), 1)
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {target_age_col}</div><div class="metric-value">{avg_val1}</div></div>', unsafe_allow_html=True)
    with kpi_col4:
        target_inc_col = next((f for f in SELECTED_FEATURES if "income" in f.lower() or "pendapatan" in f.lower() or "gaji" in f.lower()), SELECTED_FEATURES[1])
        avg_val2 = round(df_clean[target_inc_col].mean(), 1)
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {target_inc_col}</div><div class="metric-value">{avg_val2}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Profil Segmen & Rekomendasi Target")
    
    # Kelompokkan profil rata-rata per klaster
    agg_rules = {f: 'mean' for f in SELECTED_FEATURES}
    agg_rules["Cluster_ID"] = 'count'
    cluster_profiles = df_clean.groupby("Cluster_ID").agg(agg_rules).rename(columns={"Cluster_ID": "Jumlah Anggota"}).round(2)
    
    # Render kartu informasi segmen
    for idx, row in cluster_profiles.iterrows():
        p_style = get_persona_styling(idx, n_clusters)
        label_tampilan = custom_names[idx]
        persen_populasi = round((row['Jumlah Anggota'] / len(df_clean)) * 100, 1)
        
        feature_stats = ""
        for f in SELECTED_FEATURES:
            feature_stats += f'<div class="stat-item"><div class="stat-label">Rerata {f}</div><div class="stat-val">{row[f]}</div></div>'
            
        card_html = f"""
        <div class="insight-card" style="border-left-color: {p_style['color']};">
            <div class="insight-header">
                <div class="insight-title">{label_tampilan} (ID: {idx+1})</div>
                <div class="insight-badge" style="background-color: {p_style['color']};">{persen_populasi}% dari Populasi</div>
            </div>
            <div class="insight-stats">
                <div class="stat-item"><div class="stat-label">Total Anggota</div><div class="stat-val">{int(row['Jumlah Anggota'])} Orang</div></div>
                {feature_stats}
            </div>
            <div class="strategy-box">
                <strong>Rekomendasi Aksi & Target:</strong> {p_style['desc']}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: DATA SCHEMAS & TUTORIAL
# -----------------------------------------------------------------------------
with tab_tutorial:
    st.markdown("### Panduan Persiapan Database CSV")
    st.markdown("Pastikan dataset yang Anda unggah mengikuti struktur tabel di bawah ini agar pemrosesan algoritma berjalan optimal.")
    
    col_t1, col_t2 = st.columns([1.2, 0.8])
    
    with col_t1:
        st.markdown("#### Struktur Tabel Database yang Direkomendasikan")
        
        schema_data = {
            "Nama Kolom (Header)": [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ],
            "Tipe Data": [
                "Numerik (Integer)",
                "Teks / Kategori",
                "Numerik (Integer)",
                "Numerik (Float/Integer)",
                "Numerik (Integer)"
            ],
            "Contoh Nilai": [
                "1, 2, 3...",
                "Male / Female",
                "19, 35, 54...",
                "15, 60, 137...",
                "39, 81, 5..."
            ],
            "Aturan Validasi": [
                "ID Unik Pelanggan (Tidak dimasukkan ke fitur latih ML)",
                "Opsional (Hanya untuk analisa deskriptif silang di EDA)",
                "Usia dalam tahun (minimal 10)",
                "Pendapatan tahunan dalam satuan k$ atau angka bulat",
                "Skor belanja buatan sistem (skala 1 s.d. 100)"
            ]
        }
        st.table(pd.DataFrame(schema_data))
        
        st.markdown("#### Validasi Kelayakan Data Sebelum Unggah")
        st.markdown("""
        * **Pembersihan Missing Value**: Platform ini otomatis menghapus baris yang memiliki nilai kosong pada kolom analisis yang dipilih.
        * **Normalisasi Skala Otomatis**: Normalisasi (Z-Score atau MinMax) diterapkan untuk menghindari dominasi fitur dengan nominal besar (seperti Gaji) atas fitur nominal kecil (seperti Usia).
        * **Filter Outlier**: Bila diaktifkan di sidebar, sistem akan menyaring nilai data ekstrim yang dapat menggeser sentroid K-Means.
        """)
        
    with col_t2:
        st.markdown("#### Unduh Berkas Contoh")
        st.markdown("Silakan unduh file sampel database mall customer di bawah ini sebagai acuan format:")
        
        if os.path.exists(DEFAULT_CSV):
            with open(DEFAULT_CSV, "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                label="Unduh Sampel Mall_Customers.csv",
                data=csv_bytes,
                file_name="Mall_Customers.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("File contoh default Mall_Customers.csv tidak ditemukan.")
            
        st.markdown("---")
        st.markdown("#### Alur Analisis Data Platform:")
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">1</span> Unduh file sampel CSV atau siapkan file database pribadi Anda.
        </div>
        <div class="step-card">
            <span class="step-badge">2</span> Unggah file ke sistem menggunakan area drag-and-drop di sidebar kiri.
        </div>
        <div class="step-card">
            <span class="step-badge">3</span> Tentukan metode normalisasi dan aktifkan penyaringan outlier bila diperlukan.
        </div>
        <div class="step-card">
            <span class="step-badge">4</span> Gunakan grafik sikut (Elbow Curve) di tab mesin klastering untuk mencari nilai K paling optimal.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Analisis Statistik Deskriptif")
    st.markdown("Pahami persebaran fitur dan korelasi awal dari dataset yang diunggah.")
    
    st.markdown("#### Ringkasan Statistik Dasar")
    st.dataframe(df_clean[SELECTED_FEATURES].describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### Distribusi Sebaran Fitur")
        dist_feature = st.selectbox("Variabel yang Dianalisa:", SELECTED_FEATURES, index=0)
        
        fig_hist = px.histogram(
            df_clean, x=dist_feature, marginal="box", 
            color_discrete_sequence=['#4F46E5'], opacity=0.8,
            title=f"Distribusi Fitur: {dist_feature}"
        )
        fig_hist.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF")
        fig_hist.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_hist.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Gender Analysis
        gender_col = next((col for col in df_clean.columns if "gender" in col.lower() or "kelamin" in col.lower()), None)
        if gender_col:
            st.markdown(f"#### Distribusi Berdasarkan Kategori {gender_col}")
            fig_gender = px.histogram(
                df_clean, x=dist_feature, color=gender_col, 
                marginal="rug", barmode="overlay",
                color_discrete_sequence=['#4F46E5', '#EC4899']
            )
            fig_gender.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_gender, use_container_width=True)
        
    with eda_col2:
        st.markdown("#### Korelasi Tren Pairwise")
        
        x_axis_scatter = st.selectbox("Variabel Sumbu X:", SELECTED_FEATURES, index=min(1, len(SELECTED_FEATURES)-1))
        y_axis_scatter = st.selectbox("Variabel Sumbu Y:", SELECTED_FEATURES, index=0)
        
        color_by_col = gender_col if gender_col else None
        
        fig_scatter = px.scatter(
            df_clean, x=x_axis_scatter, y=y_axis_scatter, 
            color=color_by_col,
            color_discrete_sequence=['#4F46E5', '#EC4899'] if color_by_col else ['#10B981'],
            opacity=0.75, title=f"Korelasi Scatter: {x_axis_scatter} vs {y_axis_scatter}"
        )
        fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF")
        fig_scatter.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("#### Heatmap Matriks Korelasi")
        corr = df_clean[SELECTED_FEATURES].corr()
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto", 
            color_continuous_scale="RdBu_r", origin="lower",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: AI CLUSTERING ENGINE
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Mesin Partisi Segmentasi K-Means")
    st.markdown("Analisis optimasi pengelompokan segmen, metrik validasi, dan kustomisasi penamaan segmen.")
    
    # Kustomisasi Nama Label Segmen
    st.markdown("#### Penamaan Kustom Label Segmen")
    st.caption("Ubah nama segmen di bawah untuk merefleksikan istilah bisnis Anda. Nama baru akan otomatis diperbarui ke seluruh grafik, simulator, dan ekspor.")
    
    # Gunakan form input kolom
    renamer_cols = st.columns(min(3, n_clusters))
    for idx in range(n_clusters):
        col_idx = idx % 3
        with renamer_cols[col_idx]:
            st.text_input(
                label=f"Nama Segmen ID {idx+1}", 
                key=f"cname_{idx}"
            )
            
    st.markdown("---")
    
    # Elbow Method Plot
    st.markdown("#### Evaluasi Jumlah Klaster Optimal (Elbow Method)")
    
    # Hitung WCSS dinamis
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km_test = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=random_seed, n_init=10)
        km_test.fit(X_scaled)
        wcss.append(km_test.inertia_)
        
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range), y=wcss, 
        mode='lines+markers',
        line=dict(color='#4F46E5', width=3),
        marker=dict(size=8, color='#0F172A', symbol='square')
    ))
    fig_elbow.add_vline(x=n_clusters, line_width=1.5, line_dash="dash", line_color="#F43F5E")
    fig_elbow.add_annotation(x=n_clusters, y=wcss[n_clusters-1], text=f"Pilihan Anda (K={n_clusters})", showarrow=True, arrowhead=1, yshift=10)
    
    fig_elbow.update_layout(
        title="Kurva WCSS vs Jumlah Klaster (Elbow Curve)",
        xaxis_title="Jumlah Klaster (K)",
        yaxis_title="Within-Cluster Sum of Squares (Inertia)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#FFFFFF"
    )
    fig_elbow.update_xaxes(showgrid=True, gridcolor="#F1F5F9", tickmode='linear')
    fig_elbow.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.markdown("---")
    
    # Validasi Metrik
    st.markdown("#### Metrik Kualitas Pemisahan Klaster")
    mv_1, mv_2, mv_3 = st.columns(3)
    mv_1.metric(
        label="Silhouette Score (Mendekati 1.0 Lebih Baik)", 
        value=f"{sil_score:.4f}", 
        help="Evaluasi kerapatan jarak titik ke klasternya dibanding klaster tetangga."
    )
    mv_2.metric(
        label="Davies-Bouldin Index (Mendekati 0.0 Lebih Baik)", 
        value=f"{db_score:.4f}", 
        help="Evaluasi rasio jarak dalam klaster dengan jarak antar klaster."
    )
    mv_3.metric(
        label="Calinski-Harabasz Index (Skor Tinggi Lebih Baik)", 
        value=f"{ch_score:.1f}", 
        help="Evaluasi rasio dispersi antar klaster terhadap dispersi dalam klaster."
    )
    
    st.markdown("---")
    
    # Visualisasi Klaster
    plot_col1, plot_col2 = st.columns([1.4, 1])
    
    df_sorted = df_clean.sort_values(by="Cluster_ID")
    
    with plot_col1:
        if len(SELECTED_FEATURES) >= 3:
            st.markdown("#### Proyeksi Spasial 3 Dimensi")
            fig_3d = px.scatter_3d(
                df_sorted, 
                x=SELECTED_FEATURES[0], 
                y=SELECTED_FEATURES[1], 
                z=SELECTED_FEATURES[2],
                color="Cluster_Name", 
                color_discrete_map=color_map,
                hover_data=SELECTED_FEATURES
            )
            fig_3d.update_traces(marker=dict(size=4.5, line=dict(width=0.5, color='white')))
            fig_3d.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.markdown("#### Proyeksi Spasial 2 Dimensi")
            fig_2d = px.scatter(
                df_sorted, 
                x=SELECTED_FEATURES[0], 
                y=SELECTED_FEATURES[1],
                color="Cluster_Name", 
                color_discrete_map=color_map,
                hover_data=SELECTED_FEATURES
            )
            fig_2d.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
            fig_2d.update_layout(height=500, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_2d, use_container_width=True)
            
    with plot_col2:
        st.markdown("#### Reduksi Komponen Utama 2D (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PCA Component 1", "PCA Component 2"])
        df_pca["Cluster_Name"] = df_sorted["Cluster_Name"].values
        
        fig_pca = px.scatter(
            df_pca, x="PCA Component 1", y="PCA Component 2", 
            color="Cluster_Name", color_discrete_map=color_map, 
            opacity=0.85
        )
        fig_pca.update_traces(marker=dict(size=7, line=dict(width=0.5, color='white')))
        fig_pca.update_layout(height=280, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("#### Distribusi Kepadatan Klaster")
        fig_pie = px.pie(df_sorted, names="Cluster_Name", color="Cluster_Name", color_discrete_map=color_map, hole=0.5)
        fig_pie.update_layout(height=220, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Centroid Comparison Bar Chart
    st.markdown("---")
    st.markdown("#### Profil Pembanding Centroid Klaster (Skala Normalisasi)")
    st.caption("Representasi kontribusi nilai relatif dari tiap segmen. Mempermudah perbandingan segmen mana yang memiliki kriteria tinggi, sedang, atau rendah secara komprehensif.")
    
    # Buat dataframe centroid ternormalisasi
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=SELECTED_FEATURES)
    centroids_scaled["Cluster_Name"] = [custom_names[idx] for idx in range(n_clusters)]
    
    melted_centroids = pd.melt(
        centroids_scaled, 
        id_vars=["Cluster_Name"], 
        value_vars=SELECTED_FEATURES,
        var_name="Fitur", 
        value_name="Nilai Relatif (Normalized)"
    )
    
    fig_centroid_compare = px.bar(
        melted_centroids, 
        x="Fitur", 
        y="Nilai Relatif (Normalized)", 
        color="Cluster_Name",
        barmode="group", 
        color_discrete_map=color_map
    )
    fig_centroid_compare.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=20, b=20), 
        plot_bgcolor="#FFFFFF",
        xaxis_title="",
        yaxis_title="Deviasi Skala / Skor Relatif"
    )
    fig_centroid_compare.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig_centroid_compare.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_centroid_compare, use_container_width=True)

    # Tabel Koordinat Sentroid Asli
    st.markdown("#### Pusat Gravitasi Sentroid Asli (Original Scale)")
    original_centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SELECTED_FEATURES)
    original_centroids.insert(0, "Cluster_ID", range(n_clusters))
    original_centroids["Nama Segmen"] = original_centroids["Cluster_ID"].map(custom_names)
    st.dataframe(original_centroids.drop(columns="Cluster_ID").set_index("Nama Segmen").round(3), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: PREDICTIVE SIMULATOR
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Inferensi Data Pelanggan Baru")
    st.markdown("Masukkan koordinat data pelanggan baru untuk memprediksi secara waktu-nyata ke segmen mana mereka tergolong.")
    
    with st.form("inference_form_dynamic"):
        st.markdown("Masukkan Detail Atribut Pelanggan Baru:")
        
        inputs = {}
        input_cols = st.columns(min(3, len(SELECTED_FEATURES)))
        
        for i, feature in enumerate(SELECTED_FEATURES):
            col_idx = i % 3
            with input_cols[col_idx]:
                f_min = float(df_clean[feature].min())
                f_max = float(df_clean[feature].max())
                f_mean = float(df_clean[feature].mean())
                
                inputs[feature] = st.number_input(
                    label=f"{feature} (Rentang: {int(f_min)} - {int(f_max)})",
                    min_value=f_min * 0.1,
                    max_value=f_max * 2.0,
                    value=f_mean,
                    step=(f_max - f_min) / 100.0
                )
                
        submit_infer = st.form_submit_button("Prediksi Klasifikasi Segmen", use_container_width=True)
        
        if submit_infer:
            with st.spinner("Menghitung model klasifikasi..."):
                ordered_input = [inputs[f] for f in SELECTED_FEATURES]
                input_scaled = scaler.transform([ordered_input])
                pred_cluster = kmeans.predict(input_scaled)[0]
                
                label_pred = custom_names[pred_cluster]
                p_style = get_persona_styling(pred_cluster, n_clusters)
                
                st.success(f"Komputasi berhasil. Pelanggan baru diklasifikasikan ke dalam: {label_pred}.")
                
                res_html = f"""
                <div style="background-color: #FFFFFF; padding: 20px; border-radius: 8px; border-left: 6px solid {p_style['color']}; border-right: 1px solid #E2E8F0; border-top: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #0F172A;">Hasil Prediksi: {label_pred}</h4>
                    <p style="font-size: 0.95rem; color: #475569;"><strong>Strategi Segmentasi yang Direkomendasikan:</strong><br>{p_style['desc']}</p>
                    <hr style="border: 0; border-top: 1px solid #F1F5F9;">
                    <code style="font-size: 0.8rem; color: #94A3B8;">Nilai Tensor Input Terstandarisasi: {input_scaled[0]}</code>
                </div>
                """
                st.markdown(res_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5: DATA EXPORT & REPORTS
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### Manajemen Ekspor Hasil Segmentasi")
    st.markdown("Unduh seluruh basis data yang telah terklasifikasi untuk integrasi dengan dasbor visualisasi eksternal (BI) atau laporan operasional.")
    
    # Tampilkan DataFrame
    st.dataframe(df_clean, use_container_width=True, height=380)
    
    st.markdown("#### Ekspor Database")
    csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Database Berlabel (.CSV)",
        data=csv_bytes,
        file_name="Customer_Insights_Export.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>System Version: 4.1.0-Minimalist-Enterprise | Backend Engine: scikit-learn | UI Engine: Streamlit</div>", unsafe_allow_html=True)
