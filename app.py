import os
import base64
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #F8FAFC;
        }
        
        /* Hero Banner / Header */
        .hero-banner {
            background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
            padding: 50px 40px;
            border-radius: 16px;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 10%, transparent 20%);
            background-size: 20px 20px;
            opacity: 0.3;
            z-index: 0;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 15px;
            letter-spacing: -0.5px;
            z-index: 1;
            position: relative;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            font-weight: 300;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
            z-index: 1;
            position: relative;
        }

        /* Metric Cards */
        .metric-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }
        .metric-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(0,0,0,0.05);
            border-color: #3B82F6;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2rem;
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
            transition: transform 0.2s;
        }
        .insight-card:hover {
            transform: scale(1.01);
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
            font-size: 1.4rem;
            font-weight: 700;
        }
        .insight-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
        }
        .insight-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
            background-color: #F8FAFC;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #64748B;
        }
        .stat-val {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E293B;
        }
        .strategy-box {
            background-color: #EEF2F6;
            border-left: 4px solid #2563EB;
            padding: 15px;
            border-radius: 4px;
            font-size: 0.95rem;
            color: #334155;
            line-height: 1.6;
        }

        /* Tabs Styling Override */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #E0F2FE;
            color: #0284C7;
            font-weight: 600;
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
    <div class="hero-subtitle">Platform analitik canggih bertenaga Machine Learning (K-Means) untuk mengungkap pola tersembunyi dari data pelanggan Anda, dirancang khusus untuk eksekutif bisnis dan data scientist.</div>
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

def get_download_link(df, filename="clustered_data.csv", text="Unduh Laporan CSV"):
    """
    Fungsi utilitas untuk membuat tautan unduhan file CSV dari DataFrame.
    Berguna untuk mengekspor data yang telah diproses AI.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background-color: #10B981; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 10px;">📥 {text}</a>'
    return href

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

# Validasi Kolom Numerik
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom angka untuk dapat dianalisis.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Parameter Machine Learning")

# Pemilihan Kolom Pintar
# Fungsi cerdas AI untuk menebak otomatis nama kolom yang benar
def tebak_kolom(kolom_list, kata_kunci, default_idx):
    for i, nama_kolom in enumerate(kolom_list):
        if kata_kunci.lower() in nama_kolom.lower():
            return i
    return default_idx if default_idx < len(kolom_list) else 0

# Sistem sekarang akan mencari kata 'age', 'income', dan 'score'
col_age = st.sidebar.selectbox("Pilih Kolom Usia", numeric_cols, index=tebak_kolom(numeric_cols, "age", 1))
col_income = st.sidebar.selectbox("Pilih Kolom Pendapatan", numeric_cols, index=tebak_kolom(numeric_cols, "income", 2))
col_spending = st.sidebar.selectbox("Pilih Kolom Skor Belanja", numeric_cols, index=tebak_kolom(numeric_cols, "score", 3))

FEATURE_COLUMNS = [col_age, col_income, col_spending]

# Menghapus baris yang memiliki nilai NaN pada kolom fitur yang dipilih
df_clean = df_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
X = df_clean[FEATURE_COLUMNS]

# Parameter Algoritma
st.sidebar.markdown("**Tuning Algoritma K-Means**")
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
df_clean["Cluster_Name"] = "Tipe " + (df_clean["Cluster_ID"] + 1).astype(str)

st.session_state['model_trained'] = True
st.session_state['df_processed'] = df_clean

# Perhitungan Metrik Evaluasi Kinerja Model
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

# Kamus Persona Bisnis (Diasumsikan K=5 optimal untuk penjelasan, dinamis jika lebih)
PERSONA_DB = {
    0: {"name": "Premium / High Rollers", "color": "#EF4444", "icon": "👑", "desc": "Pendapatan tinggi, belanja tinggi. Berikan VIP treatment."},
    1: {"name": "Conservative / Savers", "color": "#3B82F6", "icon": "🏦", "desc": "Pendapatan tinggi, belanja rendah. Promosikan nilai investasi."},
    2: {"name": "Standard / Loyalists", "color": "#10B981", "icon": "🤝", "desc": "Pendapatan menengah, belanja stabil. Jaga dengan loyalty point."},
    3: {"name": "Impulse / Bargain Hunters", "color": "#F59E0B", "icon": "🏷️", "desc": "Pendapatan rendah, belanja tinggi. Gunakan diskon & flash sale."},
    4: {"name": "Passive / Needs-Only", "color": "#64748B", "icon": "🛒", "desc": "Pendapatan rendah, belanja rendah. Tawarkan produk primer murah."}
}

def get_persona_info(cluster_id):
    """Mengembalikan data persona berdasarkan ID, dengan fallback otomatis jika K > 5"""
    if cluster_id in PERSONA_DB:
        return PERSONA_DB[cluster_id]
    else:
        return {"name": f"Segmen Dinamis {cluster_id+1}", "color": "#0F172A", "icon": "🧩", "desc": "Segmen hasil partisi tingkat lanjut K-Means."}

# =============================================================================
# 5. ANTARMUKA UTAMA (TABS STRUCTURE)
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "📈 Exploratory Data Analysis", 
    "🧠 AI Clustering Engine", 
    "🎯 Predictive Simulator", 
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
        st.markdown(f'<div class="metric-container"><div class="metric-title">Jumlah Segmen</div><div class="metric-value">{n_clusters}</div></div>', unsafe_allow_html=True)
    with kpi3:
        avg_age = int(df_clean[col_age].mean())
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rata-rata Usia</div><div class="metric-value">{avg_age} thn</div></div>', unsafe_allow_html=True)
    with kpi4:
        avg_inc = int(df_clean[col_income].mean())
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rata-rata Gaji</div><div class="metric-value">${avg_inc}k</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🎯 Profil Persona Pelanggan")
    
    # Kalkulasi sentroid untuk profil
    cluster_profiles = df_clean.groupby("Cluster_ID").agg({
        col_age: 'mean', 
        col_income: 'mean', 
        col_spending: 'mean',
        "Cluster_Name": 'count'
    }).rename(columns={"Cluster_Name": "Jumlah Orang"}).round(1)
    
    # Generate Kartu HTML untuk masing-masing klaster
    for idx, row in cluster_profiles.iterrows():
        p_info = get_persona_info(idx)
        persentase = round((row['Jumlah Orang'] / len(df_clean)) * 100, 1)
        
        card_html = f"""
        <div class="insight-card" style="border-left-color: {p_info['color']};">
            <div class="insight-header">
                <div class="insight-title">{p_info['icon']} Tipe {idx+1} : {p_info['name']}</div>
                <div class="insight-badge" style="background-color: {p_info['color']};">{persentase}% dari Populasi</div>
            </div>
            <div class="insight-stats">
                <div class="stat-item"><div class="stat-label">👥 Jumlah Member</div><div class="stat-val">{int(row['Jumlah Orang'])} Orang</div></div>
                <div class="stat-item"><div class="stat-label">🎂 Usia Rata-rata</div><div class="stat-val">{row[col_age]} Tahun</div></div>
                <div class="stat-item"><div class="stat-label">💰 Rata-rata Gaji</div><div class="stat-val">${row[col_income]}k / thn</div></div>
            </div>
            <div class="strategy-box">
                <b>🔍 Analisis & Rekomendasi:</b> {p_info['desc']} <br>
                <i>Tingkat keaktifan belanja berada di skor {row[col_spending]} dari 100.</i>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📉 Analisis Statistik Deskriptif (EDA)")
    st.markdown("Pahami distribusi data dan korelasi antar variabel sebelum model ML diterapkan.")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("**1. Distribusi Usia Pelanggan**")
        fig_hist1 = px.histogram(df_clean, x=col_age, marginal="box", color_discrete_sequence=['#3B82F6'], opacity=0.8)
        fig_hist1.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_hist1, use_container_width=True)
        
        st.markdown("**3. Distribusi Pendapatan (Income)**")
        fig_hist2 = px.histogram(df_clean, x=col_income, marginal="box", color_discrete_sequence=['#10B981'], opacity=0.8)
        fig_hist2.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_hist2, use_container_width=True)
        
    with eda_col2:
        st.markdown("**2. Deteksi Outlier (Boxplot Multi-Variabel)**")
        fig_box = px.box(df_clean[FEATURE_COLUMNS], color_discrete_sequence=['#8B5CF6'])
        fig_box.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("**4. Matriks Korelasi (Heatmap)**")
        corr = df_clean[FEATURE_COLUMNS].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
        fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.info("💡 **Insight EDA:** Grafik di atas membantu memastikan apakah ada nilai ekstrim (outliers) yang dapat merusak kualitas model, dan bagaimana hubungan linear antar fitur (korelasi).")

# -----------------------------------------------------------------------------
# TAB 3: AI CLUSTERING ENGINE (Data Science View)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🧠 Kedalaman Model Machine Learning")
    st.markdown("Visualisasi spasial hasil segmentasi algoritma K-Means dan evaluasi metrik matematis.")
    
    # 1. Metrik Evaluasi Kualitas Klaster
    st.markdown("#### 📐 Validasi Metrik Internal")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    m_col1.metric("Silhouette Score (Mendekati 1 Lebih Baik)", f"{sil_score:.4f}", help="Mengukur seberapa rapat titik dalam klaster dan sejauh apa pemisahan antar klaster.")
    m_col2.metric("Davies-Bouldin Index (Mendekati 0 Lebih Baik)", f"{db_score:.4f}", help="Mengukur rasio penyebaran dalam klaster terhadap jarak antar klaster.")
    m_col3.metric("Calinski-Harabasz Index (Makin Tinggi Makin Baik)", f"{ch_score:.1f}", help="Rasio variansi antar-klaster dibandingkan variansi dalam-klaster.")
    
    st.markdown("---")
    
    # 2. Visualisasi Peta Scatter (3D & 2D)
    plot_col1, plot_col2 = st.columns([1.5, 1])
    
    # Mapping Warna Konsisten
    color_map = {f"Tipe {k+1}": v["color"] for k, v in PERSONA_DB.items()}
    df_sorted = df_clean.sort_values(by="Cluster_ID")
    
    with plot_col1:
        st.markdown("**Peta Klaster 3-Dimensi Ruang Fitur**")
        fig_3d = px.scatter_3d(
            df_sorted, x=col_income, y=col_spending, z=col_age,
            color="Cluster_Name", color_discrete_map=color_map,
            hover_data=[col_age, col_income, col_spending]
        )
        fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
        fig_3d.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with plot_col2:
        st.markdown("**Principal Component Analysis (PCA) 2D**")
        st.caption("Mereduksi 3 fitur menjadi 2 komponen utama agar mudah dilihat secara flat (datar).")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PCA 1", "PCA 2"])
        df_pca["Cluster_Name"] = df_sorted["Cluster_Name"]
        
        fig_pca = px.scatter(
            df_pca, x="PCA 1", y="PCA 2", color="Cluster_Name", 
            color_discrete_map=color_map, opacity=0.8
        )
        fig_pca.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
        fig_pca.update_layout(height=400, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Grafik Tambahan: Pie Chart Distribusi
        st.markdown("**Porsi Populasi**")
        fig_pie = px.pie(df_sorted, names="Cluster_Name", color="Cluster_Name", color_discrete_map=color_map, hole=0.4)
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: PREDICTIVE SIMULATOR (Deployment / Inference)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🔮 Simulator Prediksi Data Baru")
    st.markdown("Fitur *Inference* waktu-nyata. Uji coba algoritma dengan memasukkan data pelanggan baru untuk mengetahui di segmen mana mereka berada.")
    
    with st.form("inference_form"):
        st.markdown("Masukkan Data Pelanggan Fiktif / Baru:")
        f_col1, f_col2, f_col3 = st.columns(3)
        
        val_age = f_col1.number_input("Umur (Tahun)", min_value=10, max_value=100, value=30, step=1)
        val_inc = f_col2.number_input("Gaji Tahunan (K$)", min_value=1, max_value=200, value=50, step=1)
        val_spend = f_col3.number_input("Skor Belanja (1-100)", min_value=1, max_value=100, value=50, step=1)
        
        submit_infer = st.form_submit_button("Lakukan Prediksi Algoritma", use_container_width=True)
        
        if submit_infer:
            with st.spinner("Memproses data pada Model K-Means..."):
                # Transformasi data masukan sesuai skala yang dipelajari saat fit()
                input_scaled = scaler.transform([[val_age, val_inc, val_spend]])
                # Prediksi Cluster
                pred_cluster = kmeans.predict(input_scaled)[0]
                
                # Fetch Persona Info
                pred_info = get_persona_info(pred_cluster)
                
                # UI Alert untuk hasil
                st.success(f"Proses komputasi selesai! Pelanggan masuk ke dalam **Tipe {pred_cluster+1}**.")
                
                res_html = f"""
                <div style="background-color: #FFFFFF; padding: 25px; border-radius: 10px; border-left: 8px solid {pred_info['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 15px;">
                    <h3 style="margin-top: 0; color: #0F172A;">{pred_info['icon']} Persona: {pred_info['name']}</h3>
                    <p style="font-size: 1.1rem; color: #475569;"><b>Strategi yang harus diterapkan:</b><br>{pred_info['desc']}</p>
                    <hr style="border: 0; border-top: 1px solid #E2E8F0;">
                    <small style="color: #94A3B8;">Nilai Numerik Tensor: {input_scaled[0]}</small>
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
    
    st.markdown(get_download_link(df_clean, filename="ML_Clustered_Customers.csv"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>Versi Aplikasi: 3.1.0-Enterprise | Engine: scikit-learn 1.x | Framework: Streamlit</div>", unsafe_allow_html=True)
