import pandas as pd

df = pd.read_csv("cleaner.csv")

# Remove specific columns
df = df.drop(columns=["Unit", "2004", "2005"])

# Save the cleaned CSV
df.to_csv("cleaner2.csv", index=False)
