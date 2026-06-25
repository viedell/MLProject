import os
import base64
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Mengabaikan peringatan (warnings) agar UI tetap bersih
warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFIGURASI HALAMAN & MANAJEMEN TEMA (UI/UX)
# =============================================================================
st.set_page_config(
    page_title="Enterprise Customer Insight AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """
    Fungsi untuk menyuntikkan custom CSS ke dalam Streamlit.
    Mengatur font, warna latar, styling card, animasi, dan komponen visual lainnya
    agar menyerupai aplikasi SaaS modern.
    """
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #F8FAFC;
        }
        
        /* Hero Banner / Header */
        .hero-banner {
            background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
            padding: 45px 35px;
            border-radius: 16px;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.08) 10%, transparent 20%);
            background-size: 20px 20px;
            opacity: 0.4;
            z-index: 0;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
            z-index: 1;
            position: relative;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            font-weight: 300;
            opacity: 0.95;
            max-width: 850px;
            margin: 0 auto;
            z-index: 1;
            position: relative;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metric-container:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.06);
            border-color: #3B82F6;
        }
        .metric-title {
            font-size: 0.85rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0F172A;
        }

        /* Persona & Insight Cards */
        .insight-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03);
            border-left: 6px solid #CBD5E1;
            transition: all 0.2s ease-in-out;
        }
        .insight-card:hover {
            transform: scale(1.008);
            box-shadow: 0 8px 18px rgba(0,0,0,0.05);
        }
        .insight-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 10px;
        }
        .insight-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #1E293B;
        }
        .insight-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
        }
        .insight-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
            background-color: #F8FAFC;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
        }
        .stat-item {
            text-align: center;
        }
        .stat-label {
            font-size: 0.78rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        .stat-val {
            font-size: 1.15rem;
            font-weight: 600;
            color: #0F172A;
            margin-top: 4px;
        }
        .strategy-box {
            background-color: #F0F4F8;
            border-left: 4px solid #3B82F6;
            padding: 15px;
            border-radius: 4px;
            font-size: 0.92rem;
            color: #334155;
            line-height: 1.6;
        }

        /* Step Card for Tutorial */
        .step-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 12px;
        }
        .step-badge {
            background-color: #3B82F6;
            color: white;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin-right: 10px;
        }

        /* Tabs Styling Override */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 52px;
            white-space: pre-wrap;
            background-color: #F1F5F9;
            border-radius: 8px 8px 0px 0px;
            gap: 2px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 500;
            color: #475569;
            border: 1px solid #E2E8F0;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            color: #2563EB;
            font-weight: 700;
            border-top: 3px solid #2563EB;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Eksekusi injeksi CSS
inject_custom_css()

# Render Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">💎 Enterprise Customer Insight AI</div>
    <div class="hero-subtitle">Platform analitik cerdas bertenaga Machine Learning (K-Means) untuk memetakan segmen pelanggan Anda. Dirancang khusus untuk mempermudah pengambilan keputusan strategis eksekutif bisnis dan data scientist.</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 2. MANAJEMEN DATA & CACHING PIPELINE
# =============================================================================

@st.cache_data(ttl=3600)
def load_and_clean_data(file_source):
    """
    Memuat dataset dari CSV, melakukan pembersihan data awal, 
    dan mengembalikan DataFrame pandas yang sudah siap pakai.
    Menggunakan st.cache_data agar komputasi tidak berulang.
    """
    try:
        df = pd.read_csv(file_source)
        # Menghapus duplikasi jika ada
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

# Inisialisasi Session State
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None

# =============================================================================
# 3. SIDEBAR KONTROL & PARAMETER MODEL
# =============================================================================
st.sidebar.markdown("## ⚙️ Konfigurasi Sistem")
st.sidebar.markdown("Unggah data Anda atau gunakan data *default* dari sistem.")

DEFAULT_CSV = "Mall_Customers.csv"
uploaded_file = st.sidebar.file_uploader("📂 Unggah Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"Data '{uploaded_file.name}' berhasil dimuat!")
else:
    if os.path.exists(DEFAULT_CSV):
        df_raw = load_and_clean_data(DEFAULT_CSV)
        st.sidebar.info("Menggunakan dataset sampel (Mall_Customers.csv).")
    else:
        st.sidebar.error("Dataset default tidak ditemukan. Harap unggah data.")
        st.stop()

# Validasi & Dapatkan Kolom Numerik
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
# Filter out customer ID columns if possible (usually containing ID in the name)
non_id_numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]
if len(non_id_numeric_cols) < 2:
    # Fallback to all numeric cols if we filtered out too much
    non_id_numeric_cols = numeric_cols

if len(non_id_numeric_cols) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom numerik untuk dianalisis.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Fitur Segmentasi AI")

# Pemilihan Kolom Dinamis (Multi-select)
st.sidebar.markdown("Pilih kolom numerik yang ingin digunakan oleh AI Clustering Engine:")
# Guess default columns based on Mall Customers structure
default_selections = []
for kw in ["age", "income", "score", "belanja", "pendapatan", "usia"]:
    for col in non_id_numeric_cols:
        if kw in col.lower() and col not in default_selections:
            default_selections.append(col)

# If guess fails, default to first 3 columns
if len(default_selections) < 2:
    default_selections = non_id_numeric_cols[:min(3, len(non_id_numeric_cols))]

SELECTED_FEATURES = st.sidebar.multiselect(
    "Pilih Fitur Kunci", 
    options=non_id_numeric_cols, 
    default=default_selections
)

if len(SELECTED_FEATURES) < 2:
    st.sidebar.warning("⚠️ Harap pilih minimal 2 kolom fitur untuk melakukan klastering.")
    st.stop()

# Menghapus baris yang memiliki nilai NaN pada kolom fitur yang dipilih
df_clean = df_raw.dropna(subset=SELECTED_FEATURES).reset_index(drop=True)
X = df_clean[SELECTED_FEATURES]

# Parameter Algoritma
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Tuning Parameter K-Means")
n_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=5, step=1)
max_iter = st.sidebar.number_input("Maksimal Iterasi", min_value=100, max_value=1000, value=300, step=50)
random_seed = st.sidebar.number_input("Random State (Seed)", value=42)

# =============================================================================
# 4. PIPELINE MACHINE LEARNING (TRAINING & EVALUATING)
# =============================================================================
# Data Scaling (Standardisasi adalah hal wajib dalam jarak Euclidean seperti K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instansiasi & Fit Model K-Means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, random_state=random_seed, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Menyimpan hasil ke DataFrame
df_clean["Cluster_ID"] = cluster_labels
df_clean["Cluster_Name"] = "Segmen " + (df_clean["Cluster_ID"] + 1).astype(str)

st.session_state['model_trained'] = True
st.session_state['df_processed'] = df_clean

# Perhitungan Metrik Evaluasi Kinerja Model
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

# Kamus Persona Bisnis (Diasumsikan K=5 optimal untuk default Mall Customers)
PERSONA_DB = {
    0: {"name": "Premium / High Rollers", "color": "#EF4444", "icon": "👑", "desc": "Pendapatan tinggi, pengeluaran tinggi. Berikan penawaran eksklusif dan layanan VIP."},
    1: {"name": "Conservative / Savers", "color": "#3B82F6", "icon": "🏦", "desc": "Pendapatan tinggi, pengeluaran rendah. Tawarkan program investasi jangka panjang atau cashback menarik."},
    2: {"name": "Standard / Loyalists", "color": "#10B981", "icon": "🤝", "desc": "Pendapatan rata-rata, pengeluaran stabil. Berikan program loyalitas berkala untuk menjaga retensi."},
    3: {"name": "Impulse / Bargain Hunters", "color": "#F59E0B", "icon": "🏷️", "desc": "Pendapatan rendah, pengeluaran tinggi. Targetkan dengan promo flash sale, diskon besar, dan penawaran berbatas waktu."},
    4: {"name": "Passive / Needs-Only", "color": "#64748B", "icon": "🛒", "desc": "Pendapatan rendah, pengeluaran rendah. Berikan promo kebutuhan pokok sehari-hari dengan harga ekonomis."}
}

def get_persona_info(cluster_id, total_k):
    """Mengembalikan data persona berdasarkan ID, dengan fallback otomatis jika K > 5 atau dataset bukan mall customers"""
    # Gunakan persona preset jika jumlah cluster 5 dan fitur yang dipilih mengandung kata kunci income & spending
    is_mall_dataset = any("income" in f.lower() or "pendapatan" in f.lower() for f in SELECTED_FEATURES)
    
    if total_k == 5 and is_mall_dataset and cluster_id in PERSONA_DB:
        return PERSONA_DB[cluster_id]
    
    # Warna dinamis
    colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#64748B", "#06B6D4"]
    color = colors[cluster_id % len(colors)]
    icons = ["🎯", "💡", "🔥", "🚀", "🌟", "⚡", "🌈", "🍀", "💎", "🧩"]
    icon = icons[cluster_id % len(icons)]
    
    return {
        "name": f"Segmen Dinamis {cluster_id+1}",
        "color": color,
        "icon": icon,
        "desc": f"Pelanggan dengan karakteristik terstandarisasi klaster {cluster_id+1}. Silakan tinjau koordinat sentroid di tab AI Clustering Engine."
    }

# =============================================================================
# 5. ANTARMUKA UTAMA (TABS STRUCTURE)
# =============================================================================
tab1, tab_tutorial, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "📖 Data Tutorial & Schema",
    "📈 Exploratory Data Analysis", 
    "🧠 AI Clustering Engine", 
    "🔮 Predictive Simulator", 
    "📁 Data Export & Reports"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY (Business View)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Ringkasan Kinerja Segmentasi")
    st.markdown("Ikhtisar demografi dan pembagian pangsa pasar pelanggan Anda berdasarkan pemrosesan AI.")
    
    # KPI Metrics menggunakan HTML kustom
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Total Pelanggan</div><div class="metric-value">{len(df_clean)}</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Jumlah Segmen (K)</div><div class="metric-value">{n_clusters}</div></div>', unsafe_allow_html=True)
    with kpi3:
        # Find age or default to first selected feature
        age_col = next((f for f in SELECTED_FEATURES if "age" in f.lower() or "usia" in f.lower()), SELECTED_FEATURES[0])
        avg_val1 = int(df_clean[age_col].mean())
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {age_col}</div><div class="metric-value">{avg_val1}</div></div>', unsafe_allow_html=True)
    with kpi4:
        # Find income or default to second selected feature
        inc_col = next((f for f in SELECTED_FEATURES if "income" in f.lower() or "pendapatan" in f.lower() or "gaji" in f.lower()), SELECTED_FEATURES[1])
        avg_val2 = round(df_clean[inc_col].mean(), 1)
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {inc_col}</div><div class="metric-value">{avg_val2}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🎯 Profil Persona Pelanggan")
    
    # Kalkulasi sentroid untuk profil
    agg_dict = {f: 'mean' for f in SELECTED_FEATURES}
    agg_dict["Cluster_Name"] = 'count'
    
    cluster_profiles = df_clean.groupby("Cluster_ID").agg(agg_dict).rename(columns={"Cluster_Name": "Jumlah Orang"}).round(2)
    
    # Generate Kartu HTML untuk masing-masing klaster
    for idx, row in cluster_profiles.iterrows():
        p_info = get_persona_info(idx, n_clusters)
        persentase = round((row['Jumlah Orang'] / len(df_clean)) * 100, 1)
        
        # Build features text for the card
        feature_stats = ""
        for f in SELECTED_FEATURES:
            feature_stats += f'<div class="stat-item"><div class="stat-label">📊 Rerata {f}</div><div class="stat-val">{row[f]}</div></div>'
            
        card_html = f"""
        <div class="insight-card" style="border-left-color: {p_info['color']};">
            <div class="insight-header">
                <div class="insight-title">{p_info['icon']} {p_info['name']}</div>
                <div class="insight-badge" style="background-color: {p_info['color']};">{persentase}% dari Populasi</div>
            </div>
            <div class="insight-stats">
                <div class="stat-item"><div class="stat-label">👥 Jumlah Anggota</div><div class="stat-val">{int(row['Jumlah Orang'])} Orang</div></div>
                {feature_stats}
            </div>
            <div class="strategy-box">
                <b>🔍 Analisis Strategis & Rekomendasi:</b> {p_info['desc']}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB: DATA TUTORIAL & SCHEMA (HOW-TO SECTION)
# -----------------------------------------------------------------------------
with tab_tutorial:
    st.markdown("### 📖 Panduan Penggunaan & Schema Database CSV")
    st.markdown("Sebelum mengunggah database Anda sendiri ke platform, harap pastikan data Anda mengikuti panduan format di bawah agar AI Clustering Engine dapat bekerja dengan akurat.")
    
    col_t1, col_t2 = st.columns([1.2, 0.8])
    
    with col_t1:
        st.markdown("#### 1. Urutan & Format Kolom yang Direkomendasikan")
        st.markdown("""
        Meskipun mesin pemrosesan kami cerdas dan memungkinkan pemilihan fitur secara dinamis, format default yang paling optimal adalah dataset transaksi/pelanggan retail dengan kolom berikut:
        """)
        
        schema_data = {
            "Nama Kolom (Header)": [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ],
            "Tipe Data": [
                "Numeric (Integer)",
                "Categorical (Text)",
                "Numeric (Integer)",
                "Numeric (Float/Integer)",
                "Numeric (Integer)"
            ],
            "Contoh Nilai": [
                "1, 2, 3...",
                "Male / Female",
                "19, 35, 54...",
                "15, 60, 137...",
                "39, 81, 5..."
            ],
            "Keterangan / Validasi": [
                "ID Unik Pelanggan (Akan dilewati otomatis oleh ML)",
                "Jenis kelamin (Dapat dianalisis pada visualisasi EDA)",
                "Usia pelanggan dalam tahun (10 - 100)",
                "Pendapatan tahunan dalam ribuan USD ($k) / Rupiah",
                "Skor belanja buatan sistem (skala 1 - 100)"
            ]
        }
        st.table(pd.DataFrame(schema_data))
        
        st.markdown("#### 2. Kriteria Kelayakan Data (Data Validation)")
        st.markdown("""
        * **Tidak Boleh Ada Missing Value (NaN/Null)** pada kolom fitur yang dipilih. Baris yang memiliki nilai kosong akan dihapus secara otomatis.
        * **Minimal 2 Kolom Numerik** wajib disertakan agar model K-Means dapat menghitung jarak geometris antarpelanggan.
        * **Skala Nilai**: Model kami telah mengintegrasikan modul standardisasi data menggunakan `StandardScaler` (Z-Score Normalization), sehingga perbedaan satuan (misal: usia puluhan tahun dan gaji puluhan ribu) akan dinormalisasi secara otomatis.
        """)
        
    with col_t2:
        st.markdown("#### 📥 Unduh Dataset Sampel")
        st.markdown("Gunakan dataset sampel standar yang telah kami siapkan untuk langsung mencoba performa platform.")
        
        # Read the file and create download button
        if os.path.exists(DEFAULT_CSV):
            with open(DEFAULT_CSV, "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                label="📥 Unduh Sampel Mall_Customers.csv",
                data=csv_bytes,
                file_name="Mall_Customers.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("File sampel tidak ditemukan.")
            
        st.markdown("---")
        st.markdown("#### 💡 Langkah Pemrosesan Data:")
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">1</span> <b>Unduh</b> sampel CSV di atas atau siapkan database Anda.
        </div>
        <div class="step-card">
            <span class="step-badge">2</span> <b>Unggah</b> file Anda melalui menu sidebar di sebelah kiri.
        </div>
        <div class="step-card">
            <span class="step-badge">3</span> <b>Pilih Fitur Kunci</b> numerik pada sidebar yang ingin dianalisis.
        </div>
        <div class="step-card">
            <span class="step-badge">4</span> <b>Tentukan K</b> menggunakan bantuan grafik Elbow Method di Tab AI Clustering Engine.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Analisis Statistik Deskriptif (EDA)")
    st.markdown("Pahami distribusi data dan korelasi antar variabel sebelum model ML diterapkan.")
    
    # 1. Summary Statistics Table
    st.markdown("#### 📊 Statistik Ringkasan Dataset")
    st.dataframe(df_clean[SELECTED_FEATURES].describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### 🎯 Distribusi Variabel Tunggal")
        dist_feature = st.selectbox("Pilih variabel untuk dianalisis:", SELECTED_FEATURES, index=0)
        
        fig_hist = px.histogram(
            df_clean, x=dist_feature, marginal="box", 
            color_discrete_sequence=['#3B82F6'], opacity=0.85,
            title=f"Distribusi dan Pencilan untuk {dist_feature}"
        )
        fig_hist.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF")
        fig_hist.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
        fig_hist.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Gender Analysis if available
        gender_col = next((col for col in df_clean.columns if "gender" in col.lower() or "kelamin" in col.lower()), None)
        if gender_col:
            st.markdown(f"#### 👥 Distribusi berdasarkan {gender_col}")
            fig_gender = px.histogram(
                df_clean, x=dist_feature, color=gender_col, 
                marginal="rug", barmode="overlay",
                color_discrete_sequence=['#3B82F6', '#EC4899']
            )
            fig_gender.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_gender, use_container_width=True)
        
    with eda_col2:
        st.markdown("#### 🔗 Hubungan Pairwise Antar Variabel (Scatter Plot)")
        
        x_axis_scatter = st.selectbox("Pilih Sumbu X Scatter:", SELECTED_FEATURES, index=min(1, len(SELECTED_FEATURES)-1))
        y_axis_scatter = st.selectbox("Pilih Sumbu Y Scatter:", SELECTED_FEATURES, index=0)
        
        color_by_col = gender_col if gender_col else None
        
        fig_scatter = px.scatter(
            df_clean, x=x_axis_scatter, y=y_axis_scatter, 
            color=color_by_col, trendline="ols",
            color_discrete_sequence=['#3B82F6', '#EC4899'] if color_by_col else ['#10B981'],
            opacity=0.75, title=f"Korelasi Tren: {x_axis_scatter} vs {y_axis_scatter}"
        )
        fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("#### 🌡️ Matriks Korelasi (Heatmap)")
        corr = df_clean[SELECTED_FEATURES].corr()
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto", 
            color_continuous_scale="RdBu_r", origin="lower",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.info("💡 **Insight EDA:** Korelasi positif kuat (mendekati 1) menunjukkan kedua fitur bergerak searah. Deteksi pencilan pada boxplot di atas penting karena K-Means sensitif terhadap nilai ekstrim.")

# -----------------------------------------------------------------------------
# TAB 3: AI CLUSTERING ENGINE (Data Science View)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🧠 Kedalaman Model Machine Learning")
    st.markdown("Visualisasi spasial hasil segmentasi algoritma K-Means, evaluasi metrik matematis, dan analisis Elbow Method.")
    
    # 1. Elbow Method Section
    st.markdown("#### 📐 Menentukan Jumlah Cluster Optimal (Elbow Method)")
    st.caption("Menghitung WCSS (Within-Cluster Sum of Squares) untuk memandu penentuan jumlah segmentasi K terbaik. Cari bagian lekukan 'sikut' di grafik.")
    
    # Precompute WCSS for K=1 to 10
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
        line=dict(color='#2563EB', width=3),
        marker=dict(size=8, color='#0F172A', symbol='circle')
    ))
    # Highlight selected K
    fig_elbow.add_vline(x=n_clusters, line_width=2, line_dash="dash", line_color="#EF4444")
    fig_elbow.add_annotation(x=n_clusters, y=wcss[n_clusters-1], text=f"Pilihan Anda (K={n_clusters})", showarrow=True, arrowhead=1, yshift=10)
    
    fig_elbow.update_layout(
        title="Kurva WCSS vs Jumlah Cluster (Elbow Curve)",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="WCSS (Inertia)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#FFFFFF"
    )
    fig_elbow.update_xaxes(showgrid=True, gridcolor="#E2E8F0", tickmode='linear')
    fig_elbow.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Metrik Evaluasi Kualitas Klaster
    st.markdown("#### ⚙️ Validasi Metrik Evaluasi Model")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    m_col1.metric(
        label="Silhouette Score (Rapat & Terpisah)", 
        value=f"{sil_score:.4f}", 
        help="Rentang -1 hingga 1. Mendekati 1 menunjukkan klaster terbentuk sangat rapat dan terpisah dari klaster lain dengan sangat baik."
    )
    m_col2.metric(
        label="Davies-Bouldin Index (Kerapatan Klaster)", 
        value=f"{db_score:.4f}", 
        help="Semakin kecil (mendekati 0) semakin baik. Menunjukkan perbandingan ukuran penyebaran klaster dengan jarak antar klaster."
    )
    m_col3.metric(
        label="Calinski-Harabasz Index (Variansi Jarak)", 
        value=f"{ch_score:.1f}", 
        help="Semakin tinggi skornya, semakin baik. Merupakan rasio jumlah dispersi antar-klaster terhadap dispersi dalam-klaster."
    )
    
    st.markdown("---")
    
    # 3. Visualisasi Peta Scatter (3D & 2D)
    plot_col1, plot_col2 = st.columns([1.4, 1])
    
    # Mapping Warna Konsisten
    color_map = {f"Segmen {k+1}": get_persona_info(k, n_clusters)["color"] for k in range(n_clusters)}
    df_sorted = df_clean.sort_values(by="Cluster_ID")
    
    with plot_col1:
        if len(SELECTED_FEATURES) >= 3:
            st.markdown("**Visualisasi 3D Ruang Fitur**")
            fig_3d = px.scatter_3d(
                df_sorted, 
                x=SELECTED_FEATURES[0], 
                y=SELECTED_FEATURES[1], 
                z=SELECTED_FEATURES[2],
                color="Cluster_Name", 
                color_discrete_map=color_map,
                hover_data=SELECTED_FEATURES
            )
            fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
            fig_3d.update_layout(height=550, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.markdown("**Visualisasi 2D Fitur yang Dipilih**")
            fig_2d = px.scatter(
                df_sorted, 
                x=SELECTED_FEATURES[0], 
                y=SELECTED_FEATURES[1],
                color="Cluster_Name", 
                color_discrete_map=color_map,
                hover_data=SELECTED_FEATURES
            )
            fig_2d.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
            fig_2d.update_layout(height=550, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_2d, use_container_width=True)
        
    with plot_col2:
        st.markdown("**Principal Component Analysis (PCA) 2D**")
        st.caption("Mereduksi seluruh dimensi fitur menjadi 2 komponen utama (PCA) untuk proyeksi 2D yang optimal.")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PCA 1", "PCA 2"])
        df_pca["Cluster_Name"] = df_sorted["Cluster_Name"]
        
        fig_pca = px.scatter(
            df_pca, x="PCA 1", y="PCA 2", color="Cluster_Name", 
            color_discrete_map=color_map, opacity=0.85
        )
        fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        fig_pca.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Porsi Populasi
        st.markdown("**Porsi Populasi per Segmen**")
        fig_pie = px.pie(df_sorted, names="Cluster_Name", color="Cluster_Name", color_discrete_map=color_map, hole=0.45)
        fig_pie.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Centroid Coordinates Table
    st.markdown("#### 📍 Koordinat Sentrol (Cluster Centroids)")
    st.caption("Pusat gravitasi (rata-rata) asli dari masing-masing klaster untuk mengartikan sifat segmen.")
    # Calculate centroids in original scale
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SELECTED_FEATURES)
    centroids.insert(0, "Cluster_ID", range(n_clusters))
    centroids["Cluster_Name"] = "Segmen " + (centroids["Cluster_ID"] + 1).astype(str)
    
    # Format table for visualization
    st.dataframe(centroids.drop(columns="Cluster_ID").set_index("Cluster_Name").round(3), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: PREDICTIVE SIMULATOR (Deployment / Inference)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🔮 Simulator Prediksi Data Baru")
    st.markdown("Fitur *Inference* waktu-nyata. Uji coba algoritma dengan memasukkan data pelanggan baru untuk mengetahui di segmen mana mereka dikelompokkan oleh AI.")
    
    with st.form("inference_form"):
        st.markdown("Masukkan Data Pelanggan Baru:")
        
        # Generate inputs dynamically based on selected features
        inputs = {}
        cols_for_inputs = st.columns(min(3, len(SELECTED_FEATURES)))
        
        for i, feature in enumerate(SELECTED_FEATURES):
            col_idx = i % 3
            with cols_for_inputs[col_idx]:
                # Find min, max, mean for initial value and boundaries
                f_min = float(df_clean[feature].min())
                f_max = float(df_clean[feature].max())
                f_mean = float(df_clean[feature].mean())
                
                inputs[feature] = st.number_input(
                    label=f"{feature} (Rentang: {int(f_min)} - {int(f_max)})",
                    min_value=f_min * 0.5,
                    max_value=f_max * 1.5,
                    value=f_mean,
                    step=(f_max - f_min) / 100.0
                )
        
        submit_infer = st.form_submit_button("Lakukan Prediksi Klaster", use_container_width=True)
        
        if submit_infer:
            with st.spinner("Memproses inferensi data baru pada model..."):
                # Urutkan input sesuai urutan fit()
                ordered_input = [inputs[f] for f in SELECTED_FEATURES]
                # Scale input
                input_scaled = scaler.transform([ordered_input])
                # Predict cluster
                pred_cluster = kmeans.predict(input_scaled)[0]
                
                # Fetch Persona Info
                pred_info = get_persona_info(pred_cluster, n_clusters)
                
                # UI Alert untuk hasil
                st.success(f"Proses komputasi selesai! Pelanggan masuk ke dalam **Segmen {pred_cluster+1}**.")
                
                res_html = f"""
                <div style="background-color: #FFFFFF; padding: 25px; border-radius: 10px; border-left: 8px solid {pred_info['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 15px;">
                    <h3 style="margin-top: 0; color: #0F172A;">{pred_info['icon']} Persona: {pred_info['name']}</h3>
                    <p style="font-size: 1.1rem; color: #475569;"><b>Strategi yang harus diterapkan:</b><br>{pred_info['desc']}</p>
                    <hr style="border: 0; border-top: 1px solid #E2E8F0;">
                    <small style="color: #94A3B8;">Nilai Koordinat Skala Terstandarisasi (Tensor): {input_scaled[0]}</small>
                </div>
                """
                st.markdown(res_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5: DATA EXPORT & REPORTS
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📁 Basis Data Klastering & Laporan Ekspor")
    st.markdown("Lihat tabel lengkap hasil pemrosesan algoritma dan unduh dalam format CSV untuk kebutuhan pelaporan bisnis atau integrasi dengan *dashboard* BI eksternal.")
    
    # Menampilkan DataFrame
    st.dataframe(df_clean, use_container_width=True, height=400)
    
    # Modul Ekspor
    st.markdown("#### Manajemen Unduhan")
    st.markdown("Klik tombol di bawah untuk mengekspor tabel di atas yang telah berisi label klasifikasi Machine Learning.")
    
    # Generate CSV in memory for download
    csv_export = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Ekspor Hasil Segmentasi CSV",
        data=csv_export,
        file_name="ML_Clustered_Customers.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>Versi Aplikasi: 4.0.0-Enterprise | Engine: scikit-learn 1.x | Framework: Streamlit</div>", unsafe_allow_html=True)
