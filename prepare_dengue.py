import pandas as pd

# Read original file
df = pd.read_csv("dengue_35_são_paulo.csv")

# Keep only required columns
df = df[
    [
        "Data",
        "Taxa de Internações por Dengue",
        "Índice de Densidade Demográfica",
        "Emissão de CH4",
        "Emissão de CO2",
        "Emissão de NO2",
        "Índice de Pobreza",
        "Índice de Urbanização",
        "Temperatura Máxima",
        "Precipitação Média Mensal",
        "Menor Umidade Relativa"
    ]
]

# Rename columns
df.columns = [
    "date",
    "dengue",
    "pop_density",
    "ch4",
    "co2",
    "no2",
    "poverty",
    "urban",
    "temp",
    "rain",
    "humidity"
]

# Save
df.to_csv("sao_paulo_dengue.csv", index=False)

print(df.head())
print(df.shape)
import pandas as pd

df = pd.read_csv("sao_paulo_dengue.csv")

print("Rows:", len(df))
print("Unique dates:", df["date"].nunique())