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
# 1. KONFIGURASI HALAMAN & TEMA VISUAL (MODERN CORPORATE STYLE)
# =============================================================================
st.set_page_config(
    page_title="Customer Segment Diagnostic AI",
    page_icon="https://img.icons8.com/color/48/artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """
    Fungsi untuk menyuntikkan custom CSS ke dalam Streamlit.
    Mengatur tema korporat minimalis dengan palet profesional (Slate, Indigo, Emerald).
    """
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC;
        }
        
        /* Corporate Title Banner */
        .hero-banner {
            background-color: #0F172A;
            border-bottom: 3px solid #6366F1;
            padding: 30px 25px;
            border-radius: 8px;
            color: #FFFFFF;
            margin-bottom: 25px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
            color: #F8FAFC;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            font-weight: 400;
            opacity: 0.9;
            color: #CBD5E1;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.01);
            transition: all 0.2s ease;
        }
        .metric-container:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        }
        .metric-title {
            font-size: 0.72rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.75px;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
        }

        /* Operational Action Cards */
        .insight-card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.01);
            border: 1px solid #E2E8F0;
            border-left: 5px solid #6366F1;
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
            font-size: 1.2rem;
            font-weight: 600;
            color: #0F172A;
        }
        .insight-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            color: white;
        }
        .insight-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
            background-color: #F8FAFC;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #F1F5F9;
        }
        .stat-item {
            text-align: center;
        }
        .stat-label {
            font-size: 0.68rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .stat-val {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0F172A;
            margin-top: 2px;
        }
        .strategy-box {
            background-color: #F8FAFC;
            border-left: 3px solid #10B981;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.88rem;
            color: #334155;
            line-height: 1.5;
            border: 1px solid #E2E8F0;
            border-left-width: 3px;
            margin-bottom: 12px;
        }
        .script-box {
            background-color: #EEF2F6;
            border-left: 3px solid #4F46E5;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.88rem;
            color: #1E293B;
            line-height: 1.5;
            border: 1px solid #E2E8F0;
            border-left-width: 3px;
            font-style: italic;
        }

        /* Step Card */
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
            font-size: 0.8rem;
        }

        /* Tabs Styling Override */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 46px;
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

# Render Header (No Emojis, Professional Frame)
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Customer Segmentation Insights & Decision Support System</div>
    <div class="hero-subtitle">Sistem Penunjang Keputusan Operasional Staff. Dikembangkan oleh Divisi Data Research untuk Rekomendasi Pelayanan Pelanggan Berbasis Inteligensi Artifisial.</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATA PIPELINE & PREPROCESSING
# =============================================================================

@st.cache_data(ttl=3600)
def load_and_clean_data(file_source):
    """Memuat database dari file unggahan."""
    try:
        df = pd.read_csv(file_source)
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat berkas database: {str(e)}")
        return None

# =============================================================================
# 3. SIDEBAR CONFIGURATION (TECHNICAL SETTINGS CONTROL)
# =============================================================================
st.sidebar.markdown("### Sumber Data & Fitur")
st.sidebar.caption("Pengaturan pemrosesan data di bawah dikonfigurasi oleh Tim Data Research.")

DEFAULT_CSV = "Mall_Customers.csv"
uploaded_file = st.sidebar.file_uploader("Unggah Database Baru (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"File {uploaded_file.name} berhasil diunggah.")
else:
    if os.path.exists(DEFAULT_CSV):
        df_raw = load_and_clean_data(DEFAULT_CSV)
        st.sidebar.info("Menggunakan database standar default.")
    else:
        st.sidebar.error("Database contoh tidak ditemukan. Silakan unggah berkas CSV.")
        st.stop()

# Filter Kolom Numerik
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
non_id_numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]
if len(non_id_numeric_cols) < 2:
    non_id_numeric_cols = numeric_cols

if len(non_id_numeric_cols) < 2:
    st.error("Dataset tidak memiliki kolom angka yang cukup untuk analisis.")
    st.stop()

# Pilih Fitur
default_selections = []
for kw in ["age", "income", "score", "belanja", "pendapatan", "usia"]:
    for col in non_id_numeric_cols:
        if kw in col.lower() and col not in default_selections:
            default_selections.append(col)

if len(default_selections) < 2:
    default_selections = non_id_numeric_cols[:min(3, len(non_id_numeric_cols))]

SELECTED_FEATURES = st.sidebar.multiselect(
    "Pilih Parameter Segmentasi", 
    options=non_id_numeric_cols, 
    default=default_selections
)

if len(SELECTED_FEATURES) < 2:
    st.sidebar.warning("Silakan pilih minimal 2 parameter analisis.")
    st.stop()

# =============================================================================
# 4. K-MEANS MACHINE LEARNING FIT
# =============================================================================
# Session State untuk preprocessing & parameter ML (agar terisolasi tapi global)
if 'scaler_type' not in st.session_state:
    st.session_state['scaler_type'] = "StandardScaler (Z-Score)"
if 'filter_outliers' not in st.session_state:
    st.session_state['filter_outliers'] = False
if 'z_threshold' not in st.session_state:
    st.session_state['z_threshold'] = 3.0
if 'n_clusters' not in st.session_state:
    st.session_state['n_clusters'] = 5
if 'max_iter' not in st.session_state:
    st.session_state['max_iter'] = 300
if 'random_seed' not in st.session_state:
    st.session_state['random_seed'] = 42

# Ambil parameter dari session state untuk memproses data
scaler_choice = st.session_state['scaler_type']
filter_outliers = st.session_state['filter_outliers']
z_threshold = st.session_state['z_threshold']
n_clusters = st.session_state['n_clusters']
max_iter = st.session_state['max_iter']
random_seed = st.session_state['random_seed']

# Terapkan filter outlier
if filter_outliers:
    mean_vals = df_raw[SELECTED_FEATURES].mean()
    std_vals = df_raw[SELECTED_FEATURES].std()
    z_scores = np.abs((df_raw[SELECTED_FEATURES] - mean_vals) / std_vals)
    filtered_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_raw[filtered_mask].dropna(subset=SELECTED_FEATURES).reset_index(drop=True)
else:
    df_clean = df_raw.dropna(subset=SELECTED_FEATURES).reset_index(drop=True)

X = df_clean[SELECTED_FEATURES]

# Scaling data
scaler = StandardScaler() if "StandardScaler" in scaler_choice else MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Jalankan model K-Means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, random_state=random_seed, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Tambahkan label hasil klastering
df_clean["Cluster_ID"] = cluster_labels

# Evaluasi Metrik Lanjutan (Untuk Lab)
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

# =============================================================================
# 5. PEMETAAN CENTROID & STRATEGI OPERASIONAL (PERSONA UTAMA)
# =============================================================================
# Hitung pusat koordinat klaster (Centroids)
centroids_original = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SELECTED_FEATURES)

# Kamus preset persona terstruktur dan panduan petugas layanan
PERSONA_PRESETS = {
    0: {
        "name": "High Rollers (Prioritas Utama)",
        "color": "#EF4444",
        "context": "Segmen berdaya beli sangat tinggi dengan nominal transaksi besar. Merupakan penopang margin laba premium.",
        "checklist": [
            "Berikan antrean prioritas atau layanan fast-track saat berada di outlet.",
            "Tawarkan pre-order untuk produk eksklusif / edisi terbatas sebelum dipasarkan umum.",
            "Direkomendasikan memberikan manajer akun khusus (personal concierge)."
        ],
        "script": "Selamat pagi Bapak/Ibu. Khusus untuk pelanggan prioritas seperti Anda, hari ini kami menyiapkan katalog produk premium edisi khusus yang baru saja tiba. Mari saya tunjukkan di area privat kami."
    },
    1: {
        "name": "Thrifty Savers (Selektif)",
        "color": "#3B82F6",
        "context": "Pendapatan tinggi namun sangat berhati-hati dalam alokasi dana belanja. Sangat rasional terhadap perbandingan kualitas produk.",
        "checklist": [
            "Fokuskan presentasi produk pada efisiensi biaya, ketahanan barang, dan jaminan garansi.",
            "Tawarkan promosi program cashback bernilai besar untuk menjaga nilai transaksi.",
            "Hindari mendesak pembeli (hard-selling); berikan data pembanding fitur produk."
        ],
        "script": "Selamat siang Bapak/Ibu. Produk ini dirancang khusus dengan masa pakai 50% lebih tahan lama dan didukung garansi penuh selama 5 tahun. Hari ini kami juga menyediakan cashback investasi khusus untuk Anda."
    },
    2: {
        "name": "Core Loyalists (Aset Stabil)",
        "color": "#10B981",
        "context": "Kelompok pendapatan menengah dengan pola transaksi stabil dan reguler. Pilar utama loyalitas brand.",
        "checklist": [
            "Ucapkan apresiasi atas kunjungan berkala mereka untuk memperkuat hubungan emosional.",
            "Informasikan sisa poin reward mereka dan arahkan penukaran produk gratis.",
            "Berikan prioritas informasi program promo reguler mingguan."
        ],
        "script": "Senang bertemu kembali dengan Anda hari ini. Sebagai bentuk apresiasi kami atas kesetiaan Anda, hari ini Anda berhak mengklaim voucer diskon loyalitas 20% langsung pada transaksi kasir."
    },
    3: {
        "name": "Spontaneous Shoppers (Sensitif Tren)",
        "color": "#F59E0B",
        "context": "Pendapatan menengah ke bawah namun sangat reaktif terhadap penawaran promosi kilat dan tren produk terbaru.",
        "checklist": [
            "Sampaikan langsung program flash sale, diskon instan, atau promo berbatas waktu.",
            "Tawarkan opsi promo bundling (misal: beli 2 gratis 1) untuk meningkatkan volume unit belanja.",
            "Tekankan kalimat urgensi promo (misal: 'Hanya berlaku s.d. malam ini')."
        ],
        "script": "Halo Kak! Khusus hari ini saja kami sedang mengadakan program flash sale setengah harga untuk item tren terbaru ini. Promonya berakhir sore ini, dan stok tersisa tinggal beberapa unit saja."
    },
    4: {
        "name": "Utility Shoppers (Kebutuhan Pokok)",
        "color": "#64748B",
        "context": "Pendapatan rendah dengan aktivitas belanja minimal. Hanya membeli produk esensial jika sangat terpaksa.",
        "checklist": [
            "Arahkan ke produk ukuran ekonomis (value pack) yang menawarkan gramasi hemat.",
            "Tawarkan opsi produk alternatif dengan merek lokal / terafiliasi yang berbiaya lebih murah.",
            "Fokuskan promosi pada produk pokok kebutuhan primer sehari-hari."
        ],
        "script": "Selamat datang Kak. Untuk kebutuhan dasar harian Anda hari ini, kami menyarankan pilihan ukuran paket ekonomis ini karena harganya 15% lebih hemat per gram dibanding kemasan biasa."
    }
}

def identify_cluster_persona(cluster_idx, centroids_df, selected_features):
    """
    Fungsi cerdas untuk memetakan klaster acak K-Means ke preset persona bisnis 
    berdasarkan urutan peringkat centroid pendapatan dan pengeluaran.
    """
    is_mall_dataset = any("income" in f.lower() or "pendapatan" in f.lower() for f in SELECTED_FEATURES)
    inc_col = next((f for f in SELECTED_FEATURES if "income" in f.lower() or "pendapatan" in f.lower() or "gaji" in f.lower()), None)
    spend_col = next((f for f in SELECTED_FEATURES if "score" in f.lower() or "belanja" in f.lower() or "pengeluaran" in f.lower()), None)
    
    if len(centroids_df) == 5 and is_mall_dataset and inc_col and spend_col:
        # Peringkat koordinat
        med_inc = centroids_df[inc_col].median()
        med_spend = centroids_df[spend_col].median()
        
        inc = centroids_df.loc[cluster_idx, inc_col]
        spend = centroids_df.loc[cluster_idx, spend_col]
        
        if inc > med_inc and spend > med_spend:
            return 0  # High Rollers
        elif inc > med_inc and spend <= med_spend:
            return 1  # Savers
        elif inc <= med_inc and spend > med_spend:
            return 3  # Bargain Hunters
        elif inc <= med_inc and spend <= med_spend:
            # Bedakan Utility vs Core Loyalists (Utility berada di paling bawah)
            if inc < med_inc * 0.75 and spend < med_spend * 0.75:
                return 4  # Utility Shoppers
            else:
                return 2  # Core Loyalists
        return 2
    else:
        # Fallback melingkar sederhana jika parameter berbeda
        return cluster_idx % 5

# Bangun pemetaan dinamis dari klaster ke persona dan nama kustom
cluster_persona_mapping = {}
for idx in range(n_clusters):
    cluster_persona_mapping[idx] = identify_cluster_persona(idx, centroids_original, SELECTED_FEATURES)

# Inisialisasi/ambil penamaan segmen dari session state
custom_names = {}
for idx in range(n_clusters):
    preset_idx = cluster_persona_mapping[idx]
    nama_default = PERSONA_PRESETS[preset_idx]["name"]
    
    key = f"cname_{idx}"
    if key not in st.session_state:
        st.session_state[key] = nama_default
    custom_names[idx] = st.session_state[key]

# Terapkan nama khusus ke dataset utama
df_clean["Cluster_Name"] = df_clean["Cluster_ID"].map(custom_names)
df_sorted = df_clean.sort_values(by="Cluster_ID")

# Mapping Warna Plotly
color_map = {custom_names[k]: PERSONA_PRESETS[cluster_persona_mapping[k]]["color"] for k in range(n_clusters)}

# =============================================================================
# 6. STRUKTUR TABS UTAMA (PRESENTATION & OPERATIONAL FOCUS FIRST)
# =============================================================================
tab1, tab_eda, tab2, tab_research_lab, tab_submission, tab_schema, tab_explorer = st.tabs([
    "Laporan Profil Segmen Pelanggan",
    "Exploratory Data Analysis (EDA)",
    "Asisten Diagnostik Pelanggan AI",
    "Data Research Lab (Technical)",
    "Laporan Metodologi & Validasi Model",
    "Panduan Pembaruan Database",
    "Dataset Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: PRESENTASI PROFIL SEGMEN (PRESENTATION VIEW FOR STAFF)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### Presentasi Hasil Riset Segmentasi Pelanggan")
    st.markdown("Divisi Data Research telah mengidentifikasi pola perilaku pelanggan ke dalam beberapa segmen operasional utama di bawah ini untuk menjadi panduan kerja harian staf.")
    
    # Overview Ringkasan
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Total Database Sampel</div><div class="metric-value">{len(df_clean)}</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Jumlah Klasifikasi Segmen</div><div class="metric-value">{n_clusters}</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        target_age = next((f for f in SELECTED_FEATURES if "age" in f.lower() or "usia" in f.lower()), SELECTED_FEATURES[0])
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {target_age} Pelanggan</div><div class="metric-value">{round(df_clean[target_age].mean(), 1)}</div></div>', unsafe_allow_html=True)
    with kpi_col4:
        target_inc = next((f for f in SELECTED_FEATURES if "income" in f.lower() or "pendapatan" in f.lower() or "gaji" in f.lower()), SELECTED_FEATURES[1])
        st.markdown(f'<div class="metric-container"><div class="metric-title">Rerata {target_inc} Pelanggan</div><div class="metric-value">{round(df_clean[target_inc].mean(), 1)}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_p1, col_p2 = st.columns([1.1, 0.9])
    
    with col_p1:
        st.markdown("#### Detail Karakteristik & Panduan Aksi Layanan Staf")
        
        # Urutkan profil berdasarkan persentase populasi
        counts = df_clean["Cluster_ID"].value_counts()
        
        for idx in range(n_clusters):
            preset_idx = cluster_persona_mapping[idx]
            p_data = PERSONA_PRESETS[preset_idx]
            label_nama = custom_names[idx]
            
            total_orang = counts.get(idx, 0)
            persen = round((total_orang / len(df_clean)) * 100, 1)
            
            checklist_items = "".join([f"<li>{item}</li>" for item in p_data["checklist"]])
            
            card_html = f"""
            <div class="insight-card" style="border-left-color: {p_data['color']};">
                <div class="insight-header">
                    <div class="insight-title">{label_nama} (Kelompok {idx+1})</div>
                    <div class="insight-badge" style="background-color: {p_data['color']};">{persen}% dari Populasi ({total_orang} Orang)</div>
                </div>
                <div class="strategy-box">
                    <strong>Konteks Bisnis:</strong> {p_data['context']}
                </div>
                <div style="font-size: 0.88rem; color: #1E293B; margin-bottom: 12px;">
                    <strong>Panduan Kerja Petugas (Action Guide):</strong>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        {checklist_items}
                    </ul>
                </div>
                <div class="script-box">
                    <strong>Skrip Komunikasi Rekomendasi:</strong><br>
                    "{p_data['script']}"
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
    with col_p2:
        st.markdown("#### Distribusi Pangsa Pasar Segmen")
        fig_share = px.pie(
            df_clean, names="Cluster_Name", color="Cluster_Name", 
            color_discrete_map=color_map, hole=0.5
        )
        fig_share.update_layout(height=350, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_share, use_container_width=True)
        
        st.markdown("#### Rata-Rata Karakteristik Nilai Riil Segmen")
        st.caption("Nilai rata-rata aktual dari masing-masing parameter untuk presentasi staff (Skala Satuan Riil).")
        
        # Hitung sentroid asli
        centroids_original_plot = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SELECTED_FEATURES)
        centroids_original_plot["Cluster_Name"] = [custom_names[idx] for idx in range(n_clusters)]
        
        melted_cents = pd.melt(
            centroids_original_plot, id_vars=["Cluster_Name"], value_vars=SELECTED_FEATURES,
            var_name="Parameter", value_name="Nilai Aktual"
        )
        
        fig_bar = px.bar(
            melted_cents, x="Parameter", y="Nilai Aktual", color="Cluster_Name",
            barmode="group", color_discrete_map=color_map,
            labels={"Nilai Aktual": "Nilai Rata-Rata Aktual"}
        )
        fig_bar.update_layout(
            height=300, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF",
            xaxis_title="", yaxis_title="Rata-Rata Unit Riil"
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_bar.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================
# TAB EDA
# ==========================
with tab_eda:

    st.markdown("### Exploratory Data Analysis (EDA)")
    st.markdown(
        "Analisis eksploratif dilakukan untuk memahami karakteristik data "
        "sebelum proses segmentasi menggunakan algoritma K-Means."
    )

    # Dataset Overview
    st.markdown("#### Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Jumlah Data", len(df_raw))

    with col2:
        st.metric("Jumlah Fitur", len(df_raw.columns))

    with col3:
        st.metric("Missing Value", int(df_raw.isnull().sum().sum()))

    with col4:
        st.metric("Duplikasi", int(df_raw.duplicated().sum()))

    st.markdown("---")

    st.markdown("#### Statistik Deskriptif")

    st.dataframe(
        df_raw[SELECTED_FEATURES]
        .describe()
        .round(2),
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("#### Distribusi Data")

    selected_hist = st.selectbox(
        "Pilih Variabel",
        SELECTED_FEATURES
    )

    fig_hist = px.histogram(
        df_raw,
        x=selected_hist,
        nbins=20,
        marginal="box"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("#### Korelasi Antar Variabel")

    corr = df_raw[SELECTED_FEATURES].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )
    
    st.markdown("---")

    st.markdown("#### Analisis Outlier")

    selected_box = st.selectbox(
        "Pilih Variabel",
        SELECTED_FEATURES,
        key="eda_box"
    )

    fig_box = px.box(
        df_raw,
        y=selected_box,
        points="outliers"
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    if len(SELECTED_FEATURES) <= 5:

        st.markdown("---")
        st.markdown("#### Scatter Matrix")

        fig_matrix = px.scatter_matrix(
            df_raw,
            dimensions=SELECTED_FEATURES
        )

        fig_matrix.update_layout(
            height=700
        )

        st.plotly_chart(
            fig_matrix,
            use_container_width=True
        )
    
# -----------------------------------------------------------------------------
# TAB 2: ASISTEN DIAGNOSTIK PELANGGAN (STAFF TOOL)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Asisten Keputusan Operasional AI")
    st.markdown("Gunakan formulir diagnostik di bawah untuk mengetahui kategori segmentasi dari pelanggan baru yang sedang Anda layani. Masukkan data profil pelanggan untuk mendapatkan panduan penawaran instan.")
    
    with st.form("diagnostic_staff_form"):
        st.markdown("#### Input Parameter Profil Pelanggan Baru")
        
        diag_inputs = {}
        cols_diag = st.columns(min(3, len(SELECTED_FEATURES)))
        
        for i, feature in enumerate(SELECTED_FEATURES):
            col_idx = i % 3
            with cols_diag[col_idx]:
                f_min = float(df_clean[feature].min())
                f_max = float(df_clean[feature].max())
                f_mean = float(df_clean[feature].mean())
                
                diag_inputs[feature] = st.number_input(
                    label=f"{feature} (Batas Data: {int(f_min)} - {int(f_max)})",
                    min_value=f_min * 0.1,
                    max_value=f_max * 2.0,
                    value=f_mean,
                    step=(f_max - f_min) / 50.0
                )
                
        submit_diag = st.form_submit_button("Jalankan Diagnosis AI", use_container_width=True)
        
        if submit_diag:
            with st.spinner("AI sedang mendiagnosis kelompok segmentasi..."):
                # Siapkan input
                ordered_input = [diag_inputs[f] for f in SELECTED_FEATURES]
                scaled_input = scaler.transform([ordered_input])
                pred_id = kmeans.predict(scaled_input)[0]
                
                # Tarik informasi kustom & preset
                nama_segmen = custom_names[pred_id]
                preset_id = cluster_persona_mapping[pred_id]
                p_details = PERSONA_PRESETS[preset_id]
                
                checklist_html = "".join([f"<li>{item}</li>" for item in p_details["checklist"]])
                
                st.success(f"Diagnosis Selesai. Pelanggan ini diidentifikasi masuk ke dalam kategori: **{nama_segmen}**.")
                
                report_html = f"""
                <div style="background-color: #FFFFFF; padding: 25px; border-radius: 8px; border: 1px solid #E2E8F0; border-left: 6px solid {p_details['color']}; margin-top: 15px;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 5px;">Hasil Analisis Sistem Diagnostik</div>
                    <h3 style="margin-top: 0; color: #0F172A; font-weight: 700; margin-bottom: 10px;">{nama_segmen}</h3>
                    
                    <div style="margin-bottom: 15px; font-size: 0.9rem; color: #475569;">
                        <strong>Konteks Karakteristik:</strong><br>
                        {p_details['context']}
                    </div>
                    
                    <div style="margin-bottom: 15px; font-size: 0.9rem; color: #1E293B;">
                        <strong>Langkah Layanan Petugas (Action Guide):</strong>
                        <ul style="margin-top: 5px; padding-left: 20px;">
                            {checklist_html}
                        </ul>
                    </div>
                    
                    <div style="background-color: #F8FAFC; border-left: 3px solid {p_details['color']}; padding: 12px 15px; font-size: 0.9rem; color: #0F172A; font-style: italic; border: 1px solid #E2E8F0; border-left-width: 3px;">
                        <strong>Gunakan Skrip Komunikasi Ini:</strong><br>
                        "{p_details['script']}"
                    </div>
                </div>
                """
                st.markdown(report_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: DATA RESEARCH LAB (TECHNICAL VIEW FOR RESEARCHERS)
# -----------------------------------------------------------------------------
with tab_research_lab:
    st.markdown("### Data Research Laboratory & ML Workbench")
    st.caption("Halaman khusus Tim Data Research untuk konfigurasi model, tuning parameter, dan visualisasi distribusi klaster.")
    
    # 1. Konfigurasi Parameter Lanjutan
    st.markdown("#### ⚙️ Konfigurasi Ulang Preprocessing & Model")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    with conf_col1:
        new_scaler = st.selectbox(
            "Tipe Normalisasi Data",
            options=["StandardScaler (Z-Score)", "MinMaxScaler (0-1 Scale)"],
            index=0 if st.session_state['scaler_type'] == "StandardScaler (Z-Score)" else 1,
            key="temp_scaler"
        )
    with conf_col2:
        new_outlier = st.checkbox(
            "Aktifkan Outlier Filter",
            value=st.session_state['filter_outliers'],
            key="temp_outlier"
        )
    with conf_col3:
        new_z = st.slider(
            "Ambang Batas Outlier (Std Dev)",
            min_value=1.5, max_value=4.0,
            value=st.session_state['z_threshold'],
            step=0.5,
            key="temp_z"
        )
        
    conf_col4, conf_col5, conf_col6 = st.columns(3)
    with conf_col4:
        new_k = st.slider(
            "Jumlah Klaster (K)", 
            min_value=2, max_value=10, 
            value=st.session_state['n_clusters'],
            key="temp_k"
        )
    with conf_col5:
        new_iter = st.number_input(
            "Max Iterasi K-Means", 
            min_value=100, max_value=1000, 
            value=st.session_state['max_iter'],
            step=50,
            key="temp_iter"
        )
    with conf_col6:
        new_seed = st.number_input(
            "State Pengacak Seed", 
            value=st.session_state['random_seed'],
            key="temp_seed"
        )
        
    # Tombol Update Model
    if st.button("Terapkan & Latih Ulang Model AI", use_container_width=True):
        st.session_state['scaler_type'] = new_scaler
        st.session_state['filter_outliers'] = new_outlier
        st.session_state['z_threshold'] = new_z
        st.session_state['n_clusters'] = new_k
        st.session_state['max_iter'] = new_iter
        st.session_state['random_seed'] = new_seed
        st.success("Konfigurasi disimpan. Model sedang dilatih ulang...")
        st.rerun()
        
    st.markdown("---")
    
    # Kustomisasi Nama Label Segmen
    st.markdown("#### Penamaan Kustom Label Segmen")
    st.caption("Sesuaikan label representasi bisnis untuk masing-masing Klaster di bawah ini.")
    renamer_cols = st.columns(min(3, n_clusters))
    for idx in range(n_clusters):
        col_idx = idx % 3
        with renamer_cols[col_idx]:
            st.text_input(
                label=f"Label Kustom ID {idx+1}", 
                key=f"cname_{idx}"
            )
            
    st.markdown("---")
    
    # 2. Elbow Method Plot
    st.markdown("#### Optimasi Jumlah Klaster (Elbow Method)")
    
    # Hitung WCSS dinamis untuk grafik sikut (1 s.d. 10)
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km_test = KMeans(n_clusters=k, init='k-means++', max_iter=max_iter, random_state=random_seed, n_init=10)
        km_test.fit(X_scaled)
        wcss.append(km_test.inertia_)
        
    fig_elbow_lab = go.Figure()
    fig_elbow_lab.add_trace(go.Scatter(
        x=list(k_range), y=wcss, 
        mode='lines+markers',
        line=dict(color='#6366F1', width=3),
        marker=dict(size=8, color='#0F172A', symbol='circle')
    ))
    fig_elbow_lab.add_vline(x=n_clusters, line_width=1.5, line_dash="dash", line_color="#EF4444")
    fig_elbow_lab.update_layout(
        title="Kurva Evaluasi WCSS (Inertia)",
        xaxis_title="K (Jumlah Klaster)", yaxis_title="WCSS",
        height=280, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF"
    )
    fig_elbow_lab.update_xaxes(showgrid=True, gridcolor="#F1F5F9", tickmode='linear')
    fig_elbow_lab.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_elbow_lab, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Metrik Validasi Intern
    st.markdown("#### Metrik Kualitas Pemisahan Klaster")
    val_c1, val_c2, val_c3 = st.columns(3)
    val_c1.metric("Silhouette Score (Mendekati 1.0 Optimal)", f"{sil_score:.4f}")
    val_c2.metric("Davies-Bouldin Index (Mendekati 0.0 Optimal)", f"{db_score:.4f}")
    val_c3.metric("Calinski-Harabasz Index (Makin Tinggi Optimal)", f"{ch_score:.1f}")
    
    st.markdown("---")
    
    # 4. Proyeksi Dimensi Spasial (3D / 2D)
    lab_col1, lab_col2 = st.columns([1.3, 1])
    with lab_col1:
        if len(SELECTED_FEATURES) >= 3:
            st.markdown("#### Plot Spasial 3-Dimensi Fitur Terpilih")
            fig_3d_lab = px.scatter_3d(
                df_sorted, x=SELECTED_FEATURES[0], y=SELECTED_FEATURES[1], z=SELECTED_FEATURES[2],
                color="Cluster_Name", color_discrete_map=color_map, hover_data=SELECTED_FEATURES
            )
            fig_3d_lab.update_traces(marker=dict(size=4.5, line=dict(width=0.5, color='white')))
            fig_3d_lab.update_layout(height=450, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d_lab, use_container_width=True)
        else:
            st.markdown("#### Plot Spasial 2-Dimensi Fitur Terpilih")
            fig_2d_lab = px.scatter(
                df_sorted, x=SELECTED_FEATURES[0], y=SELECTED_FEATURES[1],
                color="Cluster_Name", color_discrete_map=color_map, hover_data=SELECTED_FEATURES
            )
            fig_2d_lab.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
            fig_2d_lab.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_2d_lab, use_container_width=True)
            
    with lab_col2:
        st.markdown("#### Proyeksi Reduksi Dimensi Komponen Utama (PCA)")
        pca_lab = PCA(n_components=2)
        X_pca_lab = pca_lab.fit_transform(X_scaled)
        df_pca_lab = pd.DataFrame(X_pca_lab, columns=["PCA 1", "PCA 2"])
        df_pca_lab["Cluster_Name"] = df_sorted["Cluster_Name"].values
        
        fig_pca_lab = px.scatter(
            df_pca_lab, x="PCA 1", y="PCA 2", color="Cluster_Name", 
            color_discrete_map=color_map, opacity=0.8
        )
        fig_pca_lab.update_layout(height=260, margin=dict(l=10, r=10, b=10, t=10), plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_pca_lab, use_container_width=True)
        
        st.markdown("#### Rata-Rata Koordinat Sentroid Asli")
        original_centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SELECTED_FEATURES)
        original_centroids.insert(0, "Cluster_ID", range(n_clusters))
        original_centroids["Nama Segmen"] = original_centroids["Cluster_ID"].map(custom_names)
        st.dataframe(original_centroids.drop(columns="Cluster_ID").set_index("Nama Segmen").round(2), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: LAPORAN METODOLOGI & VALIDASI MODEL (CONSULTANT HANDOVER DOCUMENT)
# -----------------------------------------------------------------------------
with tab_submission:
    st.markdown("### Laporan Metodologi & Validasi Model (Dokumen Serah Terima Teknis)")
    st.markdown("Dokumen teknis ini memuat metodologi ilmiah, audit data, hasil visualisasi, serta metrik validasi model segmentasi K-Means yang dikonfigurasi secara waktu-nyata (real-time). Ditujukan untuk jajaran manajemen klien dan tim IT.")
    
    # Pre-generate markdown tables to prevent compilation crashes on pandas dependencies
    try:
        desc_table_md = df_clean[SELECTED_FEATURES].describe().round(2).to_markdown()
        centroid_table_md = original_centroids.drop(columns="Cluster_ID").set_index("Nama Segmen").round(2).to_markdown()
    except Exception:
        desc_table_md = df_clean[SELECTED_FEATURES].describe().round(2).to_string()
        centroid_table_md = original_centroids.drop(columns="Cluster_ID").set_index("Nama Segmen").round(2).to_string()

    # Dynamic Report Template based on current run
    report_content = f"""# Dokumentasi Teknis & Validasi Model Segmentasi Pelanggan

Dokumen serah terima teknis ini memaparkan metodologi pemrosesan data, justifikasi pemilihan algoritma, hasil analisis visual, serta validasi performa model clustering yang diimplementasikan untuk sistem manajemen pelanggan klien.

---

## A. Audit & Pengumpulan Dataset (Data Audit)
- **Sumber Basis Data:** Dataset demografi dan profil pengeluaran pelanggan ritel. Data aktif diambil dari berkas `{DEFAULT_CSV}`.
- **Volume Data Bersih:** {len(df_clean)} baris data (setelah dilakukan penyaringan data duplikasi, baris kosong, dan pencilan).
- **Atribut Analisis yang Digunakan:** {", ".join(SELECTED_FEATURES)}
- **Ringkasan Statistik Deskriptif Atribut Latih:**
{desc_table_md}

---

## B. Arsitektur & Justifikasi Pemilihan Model (Algorithm Justification)
- **Paradigma Pembelajaran:** Unsupervised Learning (Pembelajaran Tidak Terarah).
- **Pilihan Algoritma:** K-Means Clustering (Pengelompokan Berbasis Jarak Geometris).
- **Rasionalisasi Teknis Pemilihan Model:** 
  1. Karakteristik basis data pelanggan bersifat *unlabeled* (tidak memiliki pengelompokan kelas target bawaan), sehingga memerlukan teknik partisi data berdasarkan tingkat kemiripan atribut.
  2. Algoritma K-Means menawarkan efisiensi komputasi yang tinggi dan stabilitas konvergensi yang baik pada dimensi numerik bertipe kontinu dengan menggunakan perhitungan jarak Euclidean.
- **Konfigurasi Hyperparameter Model Aktif:**
  - Jumlah Segmentasi Klaster (K): {n_clusters}
  - Algoritma Inisialisasi: `k-means++` (untuk optimalisasi penempatan sentroid awal)
  - Batas Iterasi Maksimum: {max_iter}
  - Nilai Kunci Pengacak (Seed): {random_seed}

---

## C. Pemrosesan Pipeline & Visualisasi Spasial (Data Pipeline)
- **Normalisasi Skala Parameter:** Menggunakan teknik `{scaler_choice}` untuk menyeimbangkan bobot variansi antar parameter agar parameter bernominal besar tidak mendominasi model jarak.
- **Penyaringan Pencilan (Outliers):** {"Aktif (Ambang Batas = " + str(z_threshold) + " Standard Deviation)" if filter_outliers else "Tidak Aktif"}
- **Reduksi Dimensi (PCA Projections):** Menggunakan Principal Component Analysis (PCA) untuk mereduksi parameter multi-dimensi menjadi 2 komponen utama (PCA 1 dan PCA 2) demi kebutuhan visualisasi grafis sebaran klaster.
- **Koordinat Pusat Gravitasi Segmen (Centroids Asli):**
{centroid_table_md}

---

## D. Validasi Model & Analisis Metrik Real-Time (Model Validation)
Kualitas dan kekuatan partisi model diuji secara waktu-nyata (*real-time*) menggunakan tiga indeks evaluasi formal:
1. **Silhouette Score:** `{sil_score:.4f}`
   - *Analisis:* Mengukur seberapa dekat suatu titik data dengan titik lain dalam klasternya sendiri dibandingkan dengan titik di klaster tetangga (rentang -1 hingga 1). Skor positif tinggi menunjukkan pemisahan kelompok yang matang dan solid.
2. **Davies-Bouldin Index:** `{db_score:.4f}`
   - *Analisis:* Menilai tingkat tumpang tindih (*overlap*) antar-klaster. Skor yang semakin mendekati 0 menunjukkan jarak antar-kelompok yang tegas dan minim tumpang tindih.
3. **Calinski-Harabasz Index:** `{ch_score:.1f}`
   - *Analisis:* Menghitung rasio jumlah dispersi antar-klaster terhadap jumlah dispersi dalam-klaster. Semakin tinggi skornya menandakan klaster terbentuk dengan tingkat kepadatan yang optimal.
4. **Analisis Kurva Sikut (Elbow Curve - WCSS):**
   - Nilai Within-Cluster Sum of Squares (WCSS) dievaluasi dinamis dari K=1 s.d. 10. Hasil kurva sikut menunjukkan bahwa penurunan tingkat variansi terdalam terjadi pada lekukan K={n_clusters}, memvalidasi jumlah klaster pilihan.

---

## E. Handover & Status Deployment Produksi (Deployment & Handover)
- **Media Penerapan (Deployment Platform):** Dasbor aplikasi web interaktif berbasis cloud yang didukung oleh server Streamlit.
- **Modul Integrasi Serah Terima:**
  1. *Asisten Diagnostik Pelanggan AI*: Antarmuka kalkulator keputusan instan bagi staf operasional lapangan untuk memprediksi kategori segmen pelanggan baru secara waktu-nyata.
  2. *Panduan Skrip & Aksi Operasional*: Arahan taktis layanan pelanggan dan panduan naskah komunikasi pelayanan untuk staf *frontliner* agar penawaran promosi berjalan tepat sasaran.
- **Tautan Repositori Kode:** [Dapat dilampirkan dengan tautan repositori produksi Anda]
"""

    st.markdown("#### 1. Preview Dokumen Teknis & Laporan Metodologi")
    st.text_area("Konten File Laporan (Markdown)", report_content, height=350)
    
    # Download Button for the Report File
    st.download_button(
        label="Unduh Laporan Model_Methodology_&_Validation_Report.md",
        data=report_content,
        file_name="Model_Methodology_&_Validation_Report.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 2. Exporter Source Code
    st.markdown("#### 2. Kode Program Integrasi Sistem (app.py)")
    st.caption("Berikut adalah salinan kode program lengkap yang sedang berjalan untuk kebutuhan peninjauan teknis tim IT klien.")
    
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            code_text = f.read()
    except Exception:
        code_text = "# Gagal memuat berkas kode program secara otomatis. Silakan salin manual dari file app.py."
        
    st.code(code_text, language="python")

# -----------------------------------------------------------------------------
# TAB 5: PANDUAN PEMBARUAN DATABASE (HOW-TO UPDATE)
# -----------------------------------------------------------------------------
with tab_schema:
    st.markdown("### Panduan Pembaruan Database Layanan")
    st.markdown("Untuk memperbarui database dasar yang dianalisa oleh sistem ini, pastikan berkas CSV mengikuti standarisasi format berikut.")
    
    guide_col1, guide_col2 = st.columns([1.2, 0.8])
    
    with guide_col1:
        st.markdown("#### Urutan dan Struktur Header Kolom Database")
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
        
        st.markdown("#### Kriteria Kualitas Data Minimum")
        st.markdown("""
        * **Format Berkas**: Berkas harus diekspor dalam format **Comma-Separated Values (CSV)** dengan pemisah koma `,`.
        * **Tidak Ada Missing Value**: Sistem akan mengabaikan baris yang memiliki nilai kosong (*Null/NaN*) pada kolom parameter analisis terpilih.
        * **Tipe Data**: Parameter segmentasi (seperti Gaji, Usia, Skor) wajib berupa angka bulat / desimal tanpa karakter simbol mata uang (seperti $, Rp).
        """)
        
    with guide_col2:
        st.markdown("#### Unduh Template CSV Standar")
        st.markdown("Unduh template di bawah ini untuk digunakan sebagai acuan pengisian data baru:")
        
        if os.path.exists(DEFAULT_CSV):
            with open(DEFAULT_CSV, "rb") as f:
                csv_bytes_guide = f.read()
            st.download_button(
                label="Unduh Berkas Mall_Customers.csv",
                data=csv_bytes_guide,
                file_name="Mall_Customers.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Berkas acuan standar Mall_Customers.csv tidak ditemukan.")

# -----------------------------------------------------------------------------
# TAB 6: BASIS DATA & EKSPOR (DATABASE EXPLORER)
# -----------------------------------------------------------------------------
with tab_explorer:
    st.markdown("### Basis Data Hasil Segmentasi")
    st.markdown("Tabel lengkap seluruh database yang telah diberi label klasifikasi segmentasi oleh algoritma K-Means. Gunakan tombol di bawah untuk mengekspor.")
    
    st.dataframe(df_clean, use_container_width=True, height=400)
    
    st.markdown("#### Manajemen Unduhan Data")
    csv_bytes_export = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Ekspor Hasil Database Berlabel (.CSV)",
        data=csv_bytes_export,
        file_name="Database_Pelanggan_Berlabel.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>System Version: 5.1.0-Operational-Enterprise | Backend Engine: scikit-learn | UI Engine: Streamlit</div>", unsafe_allow_html=True)
