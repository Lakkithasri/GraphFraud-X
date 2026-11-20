
import pandas as pd
from pathlib import Path

folder = Path.home() / "Downloads" / "GraphFraudpj"

for name in ["network_edges.csv", "account_profiles.csv",
             "fraud_patterns.csv", "time_series_stats.csv"]:

    file = folder / name

    print("\n" + "=" * 55)
    print("FILE:", name)

    if not file.exists():
        print("File not found!")
        continue

    df = pd.read_csv(file, nrows=5)

    print("Columns:", df.columns.tolist())
    print("\nSample records:")
    print(df.to_string(index=False))
