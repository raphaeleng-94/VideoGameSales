import pandas as pd

# Caminho do arquivo
file_path = './data/vgsales.csv'

# Lendo o arquivo CSV
df = pd.read_csv(file_path)

# Exibindo as colunas e tipos
"""print(df.dtypes)
print(df.head())"""
print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
print(f"Tipo das colunas: {df.dtypes}")
