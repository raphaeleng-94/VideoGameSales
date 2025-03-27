import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from logging import getLogger
import logfire
import logging
from logging import basicConfig

# ----------------------------------------------
# Configuração do logging
basicConfig(level=logging.INFO)

load_dotenv()

# ----------------------------------------------
# Configuração do Logfire
logfire.configure()
basicConfig(handlers=[logfire.LogfireLoggingHandler()])
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logfire.instrument_requests()
logfire.instrument_sqlalchemy()

# Configurações do banco de dados
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('POSTGRES_PORT')
DB_NAME = os.getenv('POSTGRES_DB')

# Lendo o arquivo CSV
file_path = './data/vgsales.csv'
df = pd.read_csv(file_path)

# 🚀 Verifique se todas as variáveis foram preenchidas corretamente
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    logger.error("❌ Variáveis de ambiente de conexão com o banco de dados não estão definidas corretamente!")
    exit()

# 🔥 Cria a conexão com o banco de dados
try:
    engine = create_engine(
        f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
    connection = engine.connect()
    logger.info("✅ Conexão com o banco de dados estabelecida com sucesso!")
except Exception as e:
    logger.error(f"❌ Erro ao conectar com o banco de dados: {e}")
    exit()

# Carregando dados para o banco de dados
table_name = 'vgsales'
df.to_sql(table_name, engine, if_exists='replace', index=False)

logger.info(f'Dados carregados com sucesso na tabela "{table_name}"')
