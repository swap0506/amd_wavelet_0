import pandas as pd

# read the file (no header)
df = pd.read_csv("exchange_rate.txt.gz", header=None, compression="gzip")

# add column names
df.columns = ['c1','c2','c3','c4','c5','c6','c7','c8']

# add a fake date column
df.insert(0, 'date', pd.date_range(start='1990-01-01', periods=len(df), freq='D'))

# save
df.to_csv("exchange_rate.csv", index=False)

print(df.head())
print(df.shape)