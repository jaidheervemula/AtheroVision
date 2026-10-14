
from pathlib import Path
import pandas as pd

file_path = Path("data/raw/extention of Z-Alizadeh sani dataset.xlsx")

if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {file_path.resolve()}"
    )

df = pd.read_excel(file_path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string())

print("\n--- MISSING VALUES ---")
print(df.isna().sum().to_string())
