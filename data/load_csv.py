import pandas as pd

# Caminho do arquivo
file_path = './data/vgsales.csv'

# Lendo o arquivo CSV
df = pd.read_csv(file_path)

# Exibindo as colunas e tipos
print(df.dtypes)
print(df.head())
