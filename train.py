import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from argparse import ArgumentParser

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==================
# SETUP LOGGING
# ==================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================
# ARGUMENT PARSER
# ==================
parser = ArgumentParser(description="Training K-Means Model dengan Auto-Tuning")
parser.add_argument("--data", type=str, default="Mall_Customers.csv", help="Path dataset CSV")
parser.add_argument("--out_dir", type=str, default="model", help="Folder untuk menyimpan model")
parser.add_argument("--auto_k", action="store_true", help="Otomatis cari nilai K (Cluster) terbaik berdasarkan Silhouette Score")
parser.add_argument("--k", type=int, default=5, help="Jumlah cluster (diabaikan jika --auto_k aktif)")
args = parser.parse_args()

# Konfigurasi Fitur
FEATURE_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

def main():
    logger.info("Memulai proses training...")
    
    # 1. LOAD DATASET
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logger.error(f"File {args.data} tidak ditemukan!")
        return

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Dataset tidak memiliki kolom wajib: {missing_cols}")
        return

    X = df[FEATURE_COLUMNS]

    # 2. SCALING
    logger.info("Melakukan scaling data (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. TRAINING & AUTO-TUNING
    best_k = args.k
    best_score = -1
    best_model = None

    if args.auto_k:
        logger.info("Mode Auto-K aktif. Mencari nilai Cluster (K) terbaik...")
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            logger.info(f"K = {k} | Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_model = km
        
        logger.info(f"🎯 K terbaik ditemukan: {best_k} (Score: {best_score:.4f})")
    else:
        logger.info(f"Melatih K-Means dengan K={args.k} (Manual)...")
        best_model = KMeans(n_clusters=args.k, random_state=42, n_init=10)
        best_model.fit(X_scaled)
        best_score = silhouette_score(X_scaled, best_model.labels_)

    # 4. SAVE MODEL & METADATA
    os.makedirs(args.out_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(args.out_dir, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    
    metadata = {
        "model": "K-Means Clustering",
        "optimal_k": best_k,
        "silhouette_score": best_score,
        "inertia": best_model.inertia_,
        "features_used": FEATURE_COLUMNS
    }
    
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"✅ Berhasil! Model, Scaler, dan Metadata disimpan di folder '{args.out_dir}/'")

if __name__ == "__main__":
    main()
