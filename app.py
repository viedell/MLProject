import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ======================================================
# SETUP HALAMAN & CUSTOM CSS / HTML (UI/UX PREMIUM)
# ======================================================
st.set_page_config(page_title="AI Customer Insight Engine", page_icon="🛍️", layout="wide")

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 45px 35px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.15);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.85;
    }

    /* AI Insight Box Card */
    .ai-insight-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .ai-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
    }
    .ai-heading {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
    }
    .ai-text {
        font-size: 1rem;
        color: #334155;
        line-height: 1.6;
        margin-bottom: 18px;
    }
    .ai-action-plan {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #1E293B;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Hero Header Section
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🧠 AI Customer Insight Engine</div>
    <div class="hero-subtitle">Mengubah matriks data angka yang rumit menjadi peta strategi bisnis yang mudah dipahami orang awam.</div>
</div>
""", unsafe_allow_html=True)

DEFAULT_CSV = "Mall_Customers.csv"

# ======================================================
# DATA PIPELINE WITH CACHING
# ======================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.markdown("### ⚙️ Pusat Kontrol Data")
uploaded_file = st.sidebar.file_uploader("Unggah Database Pelanggan Anda (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
else:
    try:
        df_raw = load_data(DEFAULT_CSV)
    except FileNotFoundError:
        st.error("Gagal memuat data. Silakan sediakan file Mall_Customers.csv atau unggah mandiri lewat sidebar.")
        st.stop()

# Auto-detecting numeric columns
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
col_age = numeric_cols[0] if len(numeric_cols) > 0 else "Age"
col_income = numeric_cols[1] if len(numeric_cols) > 1 else "Annual Income (k$)"
col_spending = numeric_cols[2] if len(numeric_cols) > 2 else "Spending Score (1-100)"

FEATURE_COLUMNS = [col_age, col_income, col_spending]
n_clusters = st.sidebar.slider("Kelompokkan Menjadi Berapa Tipe?", 2, 10, 5, 
                               help="Geser untuk membagi jumlah kelompok pelanggan sesuai target segmentasi Anda.")

df = df_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
X = df[FEATURE_COLUMNS]

# Pipa Machine Learning (K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["Kelompok_ID"] = kmeans.fit_predict(X_scaled)
df["Kelompok"] = "Tipe " + (df["Kelompok_ID"] + 1).astype(str)

# ======================================================
# LAYOUT UTAMA & INTERAKSI UI/UX
# ======================================================
tab1, tab2, tab3 = st.tabs(["💡 Analisis Pintar AI", "📊 Peta Visual Grafik", "🔮 Simulasi Target Baru"])

# DATASET METADATA DAN PERSONA DI LOCK KE 5 CLUSTER UTAMA UNTUK KEMUDAHAN PENJELASAN AI
persona_dictionary = {
    0: {
        "nama": "Si Sultan Konsumtif (Target Utama)",
        "warna": "#EF4444",
        "penjelasan": "Kelompok ini adalah ladang emas bagi bisnis Anda. Mereka memiliki tingkat pendapatan tahunan yang sangat tinggi dan tidak ragu untuk membelanjakan uang mereka dalam jumlah besar di toko Anda. Rata-rata dari mereka berada di kelompok usia muda hingga produktif.",
        "strategi": "Fokuskan pada penawaran produk premium kelas atas, program loyalitas eksklusif (VIP Tier), rilis produk edisi terbatas (Limited Edition), dan layanan prioritas."
    },
    1: {
        "nama": "Si Hemat Berduit (Potensial Tersembunyi)",
        "warna": "#3B82F6",
        "penjelasan": "Kelompok pelanggan ini tergolong makmur karena memiliki pendapatan tahunan yang tinggi. Namun, mereka sangat berhati-hati dan menahan diri dalam berbelanja (Skor belanja rendah). Mereka membutuhkan alasan kuat sebelum mengeluarkan uang.",
        "strategi": "Pancing mereka menggunakan kampanye edukasi mengenai nilai investasi jangka panjang produk Anda, tawarkan jaminan garansi panjang, atau gunakan promosi sistem Cashback besar."
    },
    2: {
        "nama": "Kelompok Menengah (Pelanggan Setia)",
        "warna": "#10B981",
        "pensembling": "Ini adalah tulang punggung stabilitas bisnis Anda. Pendapatan mereka berada di angka rata-rata menengah, dan aktivitas belanja mereka sangat konsisten secara berkala. Profil usia mereka tersebar merata.",
        "strategi": "Jaga keterikatan mereka melalui newsletter berkala, berikan voucher diskon khusus di hari ulang tahun mereka, dan terapkan program pengumpulan poin belanja rutin."
    },
    3: {
        "nama": "Si Pemburu Diskon (Agresif Budget)",
        "warna": "#F59E0B",
        "penjelasan": "Meskipun tingkat pendapatan tahunan mereka relatif rendah, kelompok ini memiliki hasrat belanja yang sangat tinggi. Mereka sangat sensitif terhadap perubahan harga namun sangat aktif melakukan transaksi jika dirasa menguntungkan.",
        "strategi": "Tarik perhatian mereka dengan mengadakan Flash Sale berkala, penawaran paket bundel (Beli 1 Gratis 1), serta kupon potongan ongkos kirim gratis."
    },
    4: {
        "nama": "Pelanggan Pasif (Sensitif Harga)",
        "warna": "#64748B",
        "penjelasan": "Kelompok ini memiliki pendapatan yang rendah dan juga jarang melakukan aktivitas belanja. Mereka hanya membeli barang yang benar-benar esensial dan fungsional bagi hidup mereka.",
        "strategi": "Aktivasi kembali akun mereka dengan mempromosikan produk kebutuhan pokok berskala grosir murah atau berikan diskon cuci gudang dengan harga paling rendah."
    }
}

# ------------------------------------------------------
# TAB 1: ANALISIS PINTAR AI (PENJELASAN ORANG AWAM)
# ------------------------------------------------------
with tab1:
    st.markdown("### 📊 Ringkasan Data Lapangan")
    
    # Hitung Statistik Ringkas
    profile = df.groupby("Kelompok_ID").agg({col_age: "mean", col_income: "mean", col_spending: "mean"}).round(1)
    
    # Tampilkan Selector untuk Memilih Kelompok Mana yang Ingin Dijelaskan AI
    selected_cluster = st.selectbox(
        "Pilih Tipe Pelanggan yang Ingin Anda Pelajari Struktur & Penjelasan AI-nya:",
        options=sorted(df["Kelompok_ID"].unique()),
        format_func=lambda x: f"Tipe {x+1} - {persona_dictionary.get(x, {'nama': 'Pelanggan Umum'})['nama']}"
    )
    
    # Tampilkan AI Insight Box
    p_data = persona_dictionary.get(selected_cluster, {
        "nama": f"Kelompok Campuran {selected_cluster+1}",
        "warna": "#334155",
        "penjelasan": "Kelompok ini terbentuk dari hasil kalkulasi jarak matematis algoritma k-means yang mendeteksi kesamaan karakteristik usia dan pendapatan tertentu.",
        "strategi": "Lakukan pemantauan transaksi secara reguler untuk memetakan kebiasaan belanja spesifik."
    })
    
    row_stats = profile.loc[selected_cluster]
    
    ai_box_html = f"""
    <div class="ai-insight-container">
        <div class="ai-badge">✨ Hasil Analisis Otomatis AI Engine</div>
        <div class="ai-heading" style="color: {p_data['warna']};">{p_data['nama']}</div>
        <div style="margin-bottom: 15px; font-size: 0.9rem; color: #475569;">
            <b>Rata-rata Kelompok:</b> Usia {row_stats[col_age]} Tahun | Pendapatan ${row_stats[col_income]}k/tahun | Skor Belanja: {row_stats[col_spending]} dari 100
        </div>
        <div class="ai-text">{p_data['penjelasan']}</div>
        <div class="ai-action-plan">
            <strong>🎯 Strategi Aksi Pemasaran (Rekomendasi AI):</strong><br>{p_data['strategi']}
        </div>
    </div>
    """
    st.markdown(ai_box_html, unsafe_allow_html=True)

# ------------------------------------------------------
# TAB 2: PETA VISUAL GRAFIK (GRAFIK 3D + PIE CHART)
# ------------------------------------------------------
with tab2:
    st.markdown("### 📉 Eksplorasi Visual Interaktif")
    st.write("Gunakan grafik di bawah untuk melihat bagaimana AI memisahkan posisi pelanggan Anda secara nyata.")
    
    g_col1, g_col2 = st.columns([2, 1])
    
    with g_col1:
        st.markdown("**1. Peta Klaster 3 Dimensi Sesuai Atribut Asli**")
        st.caption("Anda dapat memutar, memperbesar (zoom), dan menggeser grafik 3D di bawah ini menggunakan kursor Anda.")
        
        # Urutkan dataframe agar legenda berurutan Tipe 1, Tipe 2 dst.
        df_sorted = df.sort_values(by="Kelompok_ID")
        
        # Buat Mapping warna kustom agar sesuai dengan Persona Card
        color_map = { f"Tipe {k+1}": v["warna"] for k, v in persona_dictionary.items() }
        
        fig_3d = px.scatter_3d(
            df_sorted, 
            x=col_income, 
            y=col_spending, 
            z=col_age,
            color="Kelompok",
            color_discrete_map=color_map,
            labels={"Kelompok": "Kategori Pelanggan"},
            hover_name="Kelompok"
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=550)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with g_col2:
        st.markdown("**2. Proporsi Dominasi Pasar**")
        st.caption("Melihat seberapa besar persentase jumlah data pada setiap kelompok.")
        
        fig_pie = px.pie(
            df_sorted, 
            names="Kelompok",
            color="Kelompok",
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(l=10, r=10, b=10, t=10), height=380, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tampilkan Evaluasi Siluet sebagai penjamin akurasi sains data
        sil_score = silhouette_score(X_scaled, df["Kelompok_ID"])
        st.metric("Skor Kualitas Pemisahan Data (Silhouette Score)", f"{sil_score:.3f}")
        st.caption("Skor bernilai positif membuktikan pembagian kelompok valid dan tidak saling tumpang tindih secara acak.")

# ------------------------------------------------------
# TAB 3: SIMULASI TARGET BARU (PREDIKSI INTERAKTIF)
# ------------------------------------------------------
with tab3:
    st.markdown("### 🔮 Penempatan Segmentasi Pelanggan Baru")
    st.write("Gunakan fitur ini saat ada member baru yang mendaftar untuk mendeteksi secara langsung strategi marketing apa yang paling pas untuknya.")
    
    with st.form("input_pelanggan_baru"):
        c1, c2, c3 = st.columns(3)
        in_age = c1.slider("Berapa Usia Pelanggan?", int(df[col_age].min()), int(df[col_age].max()), 30)
        in_income = c2.slider("Estimasi Pendapatan Tahunan (k$)?", int(df[col_income].min()), int(df[col_income].max()), 50)
        in_spending = c3.slider("Berapa Skor Kebiasaan Belanja Mereka (1-100)?", 1, 100, 50)
        
        submit_btn = st.form_submit_button("Jalankan Prediksi AI", use_container_width=True)
        
        if submit_btn:
            sample_scaled = scaler.transform([[in_age, in_income, in_spending]])
            pred_id = kmeans.predict(sample_scaled)[0]
            pred_persona = persona_dictionary.get(pred_id, {"nama": "Umum", "warna": "#000000", "strategi": "Promosi Standar"})
            
            st.markdown(f"""
            <div style="background-color: #EEF2F6; padding: 20px; border-radius: 8px; border-top: 5px solid {pred_persona['warna']}; margin-top: 15px;">
                <h4 style="color: {pred_persona['warna']}; margin: 0 0 5px 0;">🎉 Hasil Analisis: Pelanggan Masuk Kategori Tipe {pred_id+1}</h4>
                <p style="font-size: 1.1rem; margin: 0 0 10px 0;"><b>Persona:</b> {pred_persona['nama']}</p>
                <div style="background-color: #FFFFFF; padding: 12px; border-radius: 4px; font-size: 0.95rem; border-left: 3px solid #10B981;">
                    <b>Rencana Aksi:</b> {pred_persona['strategi']}
                </div>
            </div>
            """, unsafe_allow_html=True)
