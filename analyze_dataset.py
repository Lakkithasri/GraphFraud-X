
import pandas as pd
from pathlib import Path

folder = Path.home() / "Downloads" / "GraphFraudpj"
file = folder / "transactions.csv"

total_rows = 0
fraud_counts = {}
pattern_counts = {}

print("Analyzing transaction labels...")

for chunk in pd.read_csv(file, chunksize=100_000):
    total_rows += len(chunk)

    fraud_counts_chunk = chunk["is_fraud"].value_counts(
        dropna=False
    )

    for label, count in fraud_counts_chunk.items():
        fraud_counts[label] = fraud_counts.get(label, 0) + int(count)

    patterns = chunk["fraud_pattern"].fillna("Missing")
    for pattern, count in patterns.value_counts().items():
        pattern_counts[pattern] = (
            pattern_counts.get(pattern, 0) + int(count)
        )

print("\nTotal transactions:", total_rows)
print("\nFraud label counts:")
for label, count in sorted(fraud_counts.items(), key=lambda x: str(x[0])):
    print(f"{label}: {count} ({count / total_rows * 100:.2f}%)")

print("\nFraud pattern counts:")
for pattern, count in sorted(pattern_counts.items(), key=lambda x: str(x[0])):
    print(f"{pattern}: {count}")

print("\nAnalysis complete!")