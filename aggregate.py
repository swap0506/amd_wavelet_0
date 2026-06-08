import pandas as pd

df = pd.read_csv("sao_paulo_dengue.csv")

# Aggregate municipalities
df = df.groupby("date").mean().reset_index()

# Save
df.to_csv("sao_paulo_dengue_agg.csv", index=False)

print(df.shape)
print(df.head())
import pandas as pd

df = pd.read_csv("sao_paulo_dengue_agg.csv")

print(df.isnull().sum())