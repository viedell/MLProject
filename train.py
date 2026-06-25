import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================
# KONFIGURASI
# ==================
# Ganti nama kolom di bawah ini kalau dataset lu beda strukturnya
CSV_PATH = "Mall_Customers.csv"
FEATURE_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
N_CLUSTERS = 5

# ==================
# LOAD DATASET
# ==================
df = pd.read_csv(CSV_PATH)

missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Kolom berikut tidak ditemukan di dataset: {missing_cols}\n"
        f"Kolom yang tersedia: {list(df.columns)}\n"
        f"Sesuaikan FEATURE_COLUMNS di train.py."
    )

X = df[FEATURE_COLUMNS]

# ==================
# SCALING
# ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================
# TRAINING KMEANS
# ==================
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# ==================
# SAVE MODEL
# ==================
joblib.dump(kmeans, "model/kmeans.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model berhasil disimpan ke folder model/")
print(f"Jumlah cluster   : {N_CLUSTERS}")
print(f"Fitur yang dipakai: {FEATURE_COLUMNS}")
print(f"Inertia (WCSS)   : {kmeans.inertia_:.2f}")