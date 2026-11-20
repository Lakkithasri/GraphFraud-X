
import pandas as pd

file = "cleaned_transactions.csv"

total_rows = 0
fraud_counts = {0: 0, 1: 0}
missing_values = 0

for chunk in pd.read_csv(file, chunksize=100_000):

    total_rows += len(chunk)
    missing_values += chunk.isnull().sum().sum()

    counts = chunk["is_fraud"].value_counts()

    for label, count in counts.items():
        if label in fraud_counts:
            fraud_counts[label] += int(count)

print("\n--- Final Verification ---")
print("Total rows:", total_rows)
print("Total columns:", len(chunk.columns))
print("Missing values:", missing_values)
print("Legitimate transactions:", fraud_counts[0])
print("Fraudulent transactions:", fraud_counts[1])
print("Fraud types:", chunk["fraud_pattern"].nunique())

print("\nVerification completed.")
