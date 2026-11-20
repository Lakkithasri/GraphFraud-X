
import pandas as pd
from pathlib import Path

folder = Path.home() / "Downloads" / "GraphFraudpj"

for file in folder.glob("*.csv"):
    print("\n" + "=" * 50)
    print("FILE:", file.name)

    df = pd.read_csv(file, nrows=1000)

    print("Rows inspected:", len(df))
    print("Number of columns:", len(df.columns))
    print("Column names:", df.columns.tolist())

    print("\nMissing values in inspected rows:")
    print(df.isnull().sum().to_string())

    print("\nSample records:")
    print(df.head(3).to_string(index=False))

print("\nDataset inspection completed!")