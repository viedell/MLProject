import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ======================================================
# SETUP HALAMAN
# ======================================================
st.set_page_config(
    page_title="Sistem Analisis Pelanggan",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Sistem Pintar Pengelompokan Pelanggan")
st.markdown(
    "Aplikasi ini membantu kamu memahami tipe-tipe pelanggan berdasarkan data umur, pendapatan, dan kebiasaan belanja mereka, "
    "sehingga strategi promosi bisa lebih tepat sasaran."
)

DEFAULT_CSV = "Mall_Customers.csv"

# ======================================================
# FUNGSI CACHE (Meningkatkan Performa)
# ======================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ======================================================
# SIDEBAR: DATA & PENGATURAN
# ======================================================
st.sidebar.header("📁 Unggah Data Kamu")
uploaded_file = st.sidebar.file_uploader(
    "Punya data pelanggan sendiri? Unggah di sini (format CSV)",
    type=["csv"],
    help="Pastikan data memiliki kolom umur, pendapatan, dan skor belanja."
)

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success(f"Data '{uploaded_file.name}' siap digunakan!")
else:
    try:
        df_raw = load_data(DEFAULT_CSV)
        st.sidebar.info(f"Menggunakan data contoh dari sistem.")
    except FileNotFoundError:
        st.error("Gagal menemukan data. Silakan unggah file CSV di menu samping.")
        st.stop()

with st.expander("👀 Lihat Sekilas Data Pelanggan", expanded=False):
    st.dataframe(df_raw.head())
    st.caption(f"Total data: {len(df_raw)} pelanggan")

# Konfigurasi Kolom
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Data kamu setidaknya harus punya 2 kolom angka untuk dianalisis.")
    st.stop()

def guess_default_index(options, keyword, fallback=0):
    for i, col in enumerate(options):
        if keyword.lower() in col.lower():
            return i
    return fallback

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Kolom")
col_age = st.sidebar.selectbox("Mana kolom Umur?", numeric_cols, index=guess_default_index(numeric_cols, "age", 0))
col_income = st.sidebar.selectbox("Mana kolom Pendapatan?", numeric_cols, index=guess_default_index(numeric_cols, "income", min(1, len(numeric_cols) - 1)))
col_spending = st.sidebar.selectbox("Mana kolom Skor Belanja?", numeric_cols, index=guess_default_index(numeric_cols, "spending", min(2, len(numeric_cols) - 1)))

FEATURE_COLUMNS = [col_age, col_income, col_spending]

n_clusters = st.sidebar.slider(
    "Mau dibagi jadi berapa kelompok pelanggan?", 
    min_value=2, max_value=10, value=5,
    help="Sistem akan membagi pelanggan ke dalam jumlah kelompok ini."
)

df = df_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
X = df[FEATURE_COLUMNS]

# ======================================================
# PEMROSESAN MODEL
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["Kelompok"] = kmeans.fit_predict(X_scaled)

cluster_count = df["Kelompok"].value_counts().sort_index()
biggest_cluster = cluster_count.idxmax()
cluster_spending_mean = df.groupby("Kelompok")[col_spending].mean()
top_spending_cluster = cluster_spending_mean.idxmax()

# ======================================================
# DASHBOARD UTAMA
# ======================================================
st.markdown("### 📊 Ringkasan Bisnis")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pelanggan", f"{len(df)} Orang")
col2.metric("Jumlah Kelompok", f"{n_clusters} Tipe")
col3.metric("Kelompok Terbesar", f"Tipe {biggest_cluster + 1}", f"{cluster_count[biggest_cluster]} orang")
col4.metric("Paling Suka Belanja", f"Tipe {top_spending_cluster + 1}", f"Skor {cluster_spending_mean[top_spending_cluster]:.0f}")

st.markdown("---")

# Pengurutan Tab Berdasarkan Prioritas Pengguna (Awam ke Teknis)
tab1, tab2, tab3, tab4 = st.tabs([
    "💡 Profil & Rekomendasi", 
    "🔮 Tebak Tipe Pelanggan Baru", 
    "📈 Grafik Visual", 
    "⚙️ Dapur Teknis (Evaluasi)"
])

# ------------------------------------------------------
# TAB 1: PROFIL & REKOMENDASI (User Centered)
# ------------------------------------------------------
with tab1:
    st.subheader("Rekomendasi Strategi Promosi")
    
    profile = df.groupby("Kelompok").agg({col_age: "mean", col_income: "mean", col_spending: "mean"}).round(1)
    income_median = profile[col_income].median()
    spending_median = profile[col_spending].median()

    def get_persona(age, income, spending):
        if income > income_median and spending > spending_median: return "Si Sultan Konsumtif (Target Premium)"
        elif income > income_median and spending <= spending_median: return "Si Hemat Berduit (Potensial)"
        elif income <= income_median and spending <= spending_median: return "Si Pemburu Diskon (Budget)"
        elif spending > spending_median: return "Si Loyal (Konsumtif)"
        return "Pelanggan Standar"

    recommendations = {
        "Si Sultan Konsumtif (Target Premium)": "Tawarkan produk eksklusif, VIP membership, dan barang premium.",
        "Si Hemat Berduit (Potensial)": "Pancing dengan diskon terbatas untuk barang mahal agar mereka mau keluar uang.",
        "Si Pemburu Diskon (Budget)": "Fokuskan pada promo cuci gudang, voucher gratis ongkir, dan paket hemat.",
        "Si Loyal (Konsumtif)": "Berikan program poin (loyalty reward) agar mereka terus belanja.",
        "Pelanggan Standar": "Jaga komunikasi rutin lewat email/WhatsApp dengan rekomendasi produk umum."
    }

    profile["Karakteristik"] = profile.apply(lambda row: get_persona(row[col_age], row[col_income], row[col_spending]), axis=1)

    for cluster_id, row in profile.iterrows():
        with st.container(border=True):
            st.markdown(f"#### 🏷️ Tipe {cluster_id + 1} : {row['Karakteristik']}")
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Rata-rata Umur:** {row[col_age]} thn")
            c2.write(f"**Rata-rata Gaji:** ${row[col_income]}k")
            c3.write(f"**Skor Belanja:** {row[col_spending]}")
            st.info(f"**Saran Aksi:** {recommendations[row['Karakteristik']]}")

# ------------------------------------------------------
# TAB 2: PREDIKSI (Interaktif)
# ------------------------------------------------------
with tab2:
    st.subheader("Cek Pelanggan Baru Masuk Tipe Mana?")
    st.write("Geser tombol di bawah sesuai data pelanggan barumu, lalu klik tombol prediksi.")
    
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        input_age = st.slider("Umur", int(df[col_age].min()), int(df[col_age].max()), int(df[col_age].mean()))
    with pcol2:
        input_income = st.slider("Pendapatan", int(df[col_income].min()), int(df[col_income].max()), int(df[col_income].mean()))
    with pcol3:
        input_spending = st.slider("Skor Belanja", int(df[col_spending].min()), int(df[col_spending].max()), int(df[col_spending].mean()))

    if st.button("🔍 Cek Tipe Pelanggan", type="primary"):
        sample = scaler.transform([[input_age, input_income, input_spending]])
        predicted_cluster = kmeans.predict(sample)[0]
        predicted_persona = profile.loc[predicted_cluster, "Karakteristik"]

        st.success(f"🎉 Pelanggan ini masuk ke **Tipe {predicted_cluster + 1}**")
        st.markdown(f"**Karakteristik:** {predicted_persona}")
        st.info(f"**Strategi yang cocok:** {recommendations[predicted_persona]}")

# ------------------------------------------------------
# TAB 3: VISUALISASI
# ------------------------------------------------------
with tab3:
    st.subheader("Peta Persebaran Pelanggan")
    df_plot = df.copy()
    df_plot["Kelompok_Str"] = "Tipe " + (df_plot["Kelompok"] + 1).astype(str)
    
    fig_scatter = px.scatter(
        df_plot, x=col_income, y=col_spending, color="Kelompok_Str", hover_data=[col_age],
        title="Hubungan Pendapatan dan Kebiasaan Belanja",
        labels={"Kelompok_Str": "Tipe Pelanggan"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------
# TAB 4: DAPUR TEKNIS (Disembunyikan di akhir)
# ------------------------------------------------------
with tab4:
    st.subheader("Evaluasi Kinerja Sistem (Untuk Data Scientist)")
    sil_score = silhouette_score(X_scaled, df["Kelompok"])
    st.metric("Silhouette Score (Skor Kualitas Pengelompokan)", round(sil_score, 3), 
              help="Mendekati 1 berarti pengelompokan sangat baik dan jelas pemisahannya.")
    
    st.write("Grafik Elbow (Mencari jumlah kelompok ideal):")
    wcss = []
    k_range = range(1, 11)
    for i in k_range:
        km_temp = KMeans(n_clusters=i, random_state=42, n_init=10).fit(X_scaled)
        wcss.append(km_temp.inertia_)

    elbow_df = pd.DataFrame({"K (Jumlah Kelompok)": list(k_range), "Nilai Error (WCSS)": wcss})
    fig_elbow = px.line(elbow_df, x="K (Jumlah Kelompok)", y="Nilai Error (WCSS)", markers=True)
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    st.plotly_chart(fig_elbow, use_container_width=True)
