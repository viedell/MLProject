import argparse
import numpy as np
import pandas as pd

# ==================
# ARGUMENT
# ==================
parser = argparse.ArgumentParser(
    description="Generate dummy customer dataset untuk testing app.py"
)
parser.add_argument(
    "--n", type=int, default=200,
    help="Jumlah baris data yang mau dibuat (default: 200)"
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed (kosongkan kalau mau hasil beda-beda tiap run)"
)
parser.add_argument(
    "--output", type=str, default="Mall_Customers.csv",
    help="Nama file output (default: Mall_Customers.csv)"
)
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

n = args.n

# ==================
# GENERATE DATA RANDOM
# ==================
customer_id = np.arange(1, n + 1)

gender = np.random.choice(
    ["Male", "Female"],
    size=n
)

age = np.random.randint(
    18, 71,
    size=n
)

annual_income = np.random.randint(
    15, 141,
    size=n
)

spending_score = np.random.randint(
    1, 101,
    size=n
)

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Gender": gender,
    "Age": age,
    "Annual Income (k$)": annual_income,
    "Spending Score (1-100)": spending_score
})

# ==================
# SAVE
# ==================
df.to_csv(args.output, index=False)

print(f"Dummy dataset berhasil dibuat: {args.output}")
print(f"Jumlah baris: {n}")
print(df.head())