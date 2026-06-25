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

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Smart Customer Segmentation System")
st.caption(
    "K-Means Clustering • Evaluasi Model • Persona & Rekomendasi Marketing"
)

DEFAULT_CSV = "Mall_Customers.csv"


# ======================================================
# 1. LOAD DATA (Upload atau Default)
# ======================================================

st.sidebar.header("⚙️ Konfigurasi")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset CSV kamu",
    type=["csv"]
)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Dataset diupload: {uploaded_file.name}")
else:
    try:
        df_raw = pd.read_csv(DEFAULT_CSV)
        st.sidebar.info(f"Menggunakan dataset default: {DEFAULT_CSV}")
    except FileNotFoundError:
        st.error(
            f"File '{DEFAULT_CSV}' tidak ditemukan di folder project, "
            "dan belum ada file yang diupload. Silakan upload CSV di sidebar."
        )
        st.stop()

with st.expander("📄 Preview Dataset", expanded=False):
    st.dataframe(df_raw.head(10))
    st.caption(f"Total baris: {len(df_raw)} | Total kolom: {len(df_raw.columns)}")


# ======================================================
# 2. PEMILIHAN KOLOM (Fleksibel, tidak hardcode nama kolom)
# ======================================================

st.sidebar.subheader("🧩 Pilih Kolom Fitur")

numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error(
        "Dataset minimal harus punya 2 kolom numerik untuk bisa di-cluster. "
        f"Kolom numerik yang terdeteksi: {numeric_cols}"
    )
    st.stop()


def guess_default_index(options, keyword, fallback=0):
    for i, col in enumerate(options):
        if keyword.lower() in col.lower():
            return i
    return fallback


age_guess = guess_default_index(numeric_cols, "age", 0)
income_guess = guess_default_index(
    numeric_cols, "income", min(1, len(numeric_cols) - 1)
)
spending_guess = guess_default_index(
    numeric_cols, "spending", min(2, len(numeric_cols) - 1)
)

col_age = st.sidebar.selectbox(
    "Kolom Umur (Age)", numeric_cols, index=age_guess
)
col_income = st.sidebar.selectbox(
    "Kolom Income", numeric_cols, index=income_guess
)
col_spending = st.sidebar.selectbox(
    "Kolom Spending Score", numeric_cols, index=spending_guess
)

FEATURE_COLUMNS = [col_age, col_income, col_spending]

if len(set(FEATURE_COLUMNS)) < 3:
    st.warning(
        "⚠️ Kamu memilih kolom yang sama untuk lebih dari satu fitur. "
        "Pastikan ketiga kolom (Age, Income, Spending) berbeda."
    )

n_clusters = st.sidebar.slider(
    "Jumlah Cluster (K)", min_value=2, max_value=10, value=5
)

df = df_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
X = df[FEATURE_COLUMNS]


# ======================================================
# 3. KPI DASHBOARD
# ======================================================

st.header("📊 KPI Dashboard")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_count = df["Cluster"].value_counts().sort_index()
biggest_cluster = cluster_count.idxmax()

cluster_spending_mean = df.groupby("Cluster")[col_spending].mean()
top_spending_cluster = cluster_spending_mean.idxmax()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customer", len(df))
col2.metric("Jumlah Cluster", n_clusters)
col3.metric("Cluster Terbesar", f"Cluster {biggest_cluster}",
            f"{cluster_count[biggest_cluster]} customer")
col4.metric("Spending Tertinggi", f"Cluster {top_spending_cluster}",
            f"{cluster_spending_mean[top_spending_cluster]:.1f} avg score")


# ======================================================
# 4. TABS UTAMA
# ======================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Elbow & Evaluasi",
    "🔍 Visualisasi",
    "👤 Profil & Persona",
    "💡 Insight & Rekomendasi",
    "🎯 Prediksi Customer Baru"
])


# ------------------------------------------------------
# TAB 1: ELBOW METHOD + SILHOUETTE SCORE
# ------------------------------------------------------
with tab1:

    st.subheader("1. Elbow Method")

    wcss = []
    k_range = range(1, 11)

    for i in k_range:
        km_temp = KMeans(n_clusters=i, random_state=42, n_init=10)
        km_temp.fit(X_scaled)
        wcss.append(km_temp.inertia_)

    elbow_df = pd.DataFrame({"Cluster": list(k_range), "WCSS": wcss})

    fig_elbow = px.line(
        elbow_df, x="Cluster", y="WCSS", markers=True,
        title="Elbow Method untuk Menentukan K Optimal"
    )
    fig_elbow.add_vline(
        x=n_clusters, line_dash="dash", line_color="red",
        annotation_text=f"K terpilih = {n_clusters}"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.subheader("2. Evaluasi Model — Silhouette Score")

    sil_score = silhouette_score(X_scaled, df["Cluster"])

    colA, colB = st.columns([1, 2])

    with colA:
        st.metric("Silhouette Score", round(sil_score, 3))

    with colB:
        if sil_score >= 0.7:
            label, color = "Sangat Baik", "success"
        elif sil_score >= 0.5:
            label, color = "Baik", "success"
        elif sil_score >= 0.25:
            label, color = "Cukup", "warning"
        else:
            label, color = "Buruk", "error"

        getattr(st, color)(
            f"Interpretasi: **{label}** "
            f"(Score {sil_score:.3f}). "
            "Skala: 0.7–1.0 Sangat Baik, 0.5–0.7 Baik, "
            "0.25–0.5 Cukup, <0.25 Buruk."
        )

    st.caption(
        "Silhouette Score mengukur seberapa baik data terpisah antar cluster. "
        "Semakin mendekati 1, semakin jelas pemisahan antar cluster."
    )


# ------------------------------------------------------
# TAB 2: VISUALISASI (Scatter, Distribusi, Heatmap)
# ------------------------------------------------------
with tab2:

    st.subheader("1. Scatter Plot Cluster")

    fig_scatter = px.scatter(
        df,
        x=col_income,
        y=col_spending,
        color=df["Cluster"].astype(str),
        hover_data=[col_age],
        title=f"{col_income} vs {col_spending} per Cluster",
        labels={"color": "Cluster"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("2. Distribusi Jumlah Customer per Cluster")
    st.bar_chart(cluster_count)

    st.subheader("3. Heatmap Korelasi Antar Fitur")

    fig_heat, ax = plt.subplots(figsize=(6, 4))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_heat)


# ------------------------------------------------------
# TAB 3: PROFIL CLUSTER + PERSONA + RADAR CHART
# ------------------------------------------------------
with tab3:

    st.subheader("1. Cluster Profile")

    profile = (
        df.groupby("Cluster")
        .agg({
            col_age: "mean",
            col_income: "mean",
            col_spending: "mean"
        })
        .round(2)
    )

    def get_persona(age, income, spending, income_median, spending_median):
        if income > income_median and spending > spending_median:
            return "Young Premium Shopper"
        elif income > income_median and spending <= spending_median:
            return "Potential Customer"
        elif income <= income_median and spending <= spending_median:
            return "Budget Customer"
        elif spending > spending_median:
            return "Loyal Customer"
        return "Regular Customer"

    income_median = profile[col_income].median()
    spending_median = profile[col_spending].median()

    profile["Persona"] = profile.apply(
        lambda row: get_persona(
            row[col_age], row[col_income], row[col_spending],
            income_median, spending_median
        ),
        axis=1
    )

    st.dataframe(profile, use_container_width=True)

    st.subheader("2. Radar Chart per Cluster")

    radar_norm = (profile[FEATURE_COLUMNS] - profile[FEATURE_COLUMNS].min()) / (
        profile[FEATURE_COLUMNS].max() - profile[FEATURE_COLUMNS].min() + 1e-9
    )

    fig_radar = go.Figure()

    for cluster_id in radar_norm.index:
        values = radar_norm.loc[cluster_id].tolist()
        values.append(values[0])
        categories = FEATURE_COLUMNS + [FEATURE_COLUMNS[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=f"Cluster {cluster_id}"
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Perbandingan Karakteristik Tiap Cluster (Normalized)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ------------------------------------------------------
# TAB 4: AUTO INSIGHT + RANKING + REKOMENDASI MARKETING
# ------------------------------------------------------
with tab4:

    recommendations = {
        "Young Premium Shopper":
            "Promosikan produk premium dan program loyalty reward.",
        "Potential Customer":
            "Berikan diskon tertarget untuk meningkatkan spending.",
        "Budget Customer":
            "Fokus pada promosi dan voucher hemat.",
        "Loyal Customer":
            "Berikan membership benefit dan penawaran eksklusif.",
        "Regular Customer":
            "Jaga engagement dengan kampanye yang personal."
    }

    st.subheader("1. Marketing Recommendation per Cluster")

    for cluster_id, row in profile.iterrows():
        st.markdown(f"**Cluster {cluster_id} — {row['Persona']}**")
        st.write(
            f"Rata-rata umur: {row[col_age]:.1f} | "
            f"Rata-rata {col_income}: {row[col_income]:.1f} | "
            f"Rata-rata {col_spending}: {row[col_spending]:.1f}"
        )
        st.success(recommendations[row["Persona"]])
        st.divider()

    st.subheader("2. Cluster Ranking")

    rank_largest = cluster_count.idxmax()
    rank_top_income = profile[col_income].idxmax()
    rank_top_spending = profile[col_spending].idxmax()

    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("🏆 Largest Cluster", f"Cluster {rank_largest}")
    rcol2.metric("💰 Top Income Cluster", f"Cluster {rank_top_income}")
    rcol3.metric("🔥 Top Spending Cluster", f"Cluster {rank_top_spending}")

    st.subheader("3. Auto-Generated Insight")

    dominant_pct = (cluster_count[rank_largest] / len(df)) * 100

    insight_text = f"""
    Mayoritas pelanggan ({dominant_pct:.1f}% dari total) berada pada **Cluster {rank_largest}**,
    dengan persona **{profile.loc[rank_largest, 'Persona']}**.

    - Cluster dengan income tertinggi adalah **Cluster {rank_top_income}**
      (rata-rata {col_income}: {profile.loc[rank_top_income, col_income]:.1f}).
    - Cluster dengan spending tertinggi adalah **Cluster {rank_top_spending}**
      (rata-rata {col_spending}: {profile.loc[rank_top_spending, col_spending]:.1f}).

    Rekomendasi utama: prioritaskan strategi marketing pada **Cluster {rank_top_spending}**
    sebagai target produk premium, sambil tetap menjaga retensi pada cluster dengan
    jumlah customer terbesar (**Cluster {rank_largest}**).
    """

    st.info(insight_text)

    st.subheader("4. Download Hasil Clustering")

    csv_result = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download customer_clustered.csv",
        data=csv_result,
        file_name="customer_clustered.csv",
        mime="text/csv"
    )


# ------------------------------------------------------
# TAB 5: PREDIKSI CUSTOMER BARU
# ------------------------------------------------------
with tab5:

    st.subheader("Masukkan Data Customer Baru")

    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        input_age = st.slider(
            col_age,
            int(df[col_age].min()), int(df[col_age].max()),
            int(df[col_age].mean())
        )
    with pcol2:
        input_income = st.slider(
            col_income,
            int(df[col_income].min()), int(df[col_income].max()),
            int(df[col_income].mean())
        )
    with pcol3:
        input_spending = st.slider(
            col_spending,
            int(df[col_spending].min()), int(df[col_spending].max()),
            int(df[col_spending].mean())
        )

    if st.button("🔮 Predict Cluster", type="primary"):

        sample = scaler.transform([[input_age, input_income, input_spending]])
        predicted_cluster = kmeans.predict(sample)[0]
        predicted_persona = profile.loc[predicted_cluster, "Persona"]

        st.success(f"Customer ini masuk ke **Cluster {predicted_cluster}**")
        st.info(f"Persona: **{predicted_persona}**")
        st.success(recommendations[predicted_persona])