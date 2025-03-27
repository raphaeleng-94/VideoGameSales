import subprocess
import time
import logging
from logging import basicConfig, getLogger

# ----------------------------------------------
# Configuração do Logfire
import logfire
logfire.configure()
basicConfig(handlers=[logfire.LogfireLoggingHandler()])
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logfire.instrument_requests()
logfire.instrument_sqlalchemy()

# ----------------------------------------------
# Configuração do logging
basicConfig(level=logging.INFO)

def run_script(script_name):
    logger.info(f"🔹 Executando {script_name}...")
    start_time = time.time()
    try:
        subprocess.run(["python", f"./scr/{script_name}"], check=True)
        logger.info(f"✅ {script_name} executado com sucesso em {time.time() - start_time:.2f} segundos.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao executar {script_name}: {e}")

def main():
    logger.info("🚀 Iniciando Pipeline de Vendas de Video Games...")
    
    # 1. Carregar dados para o banco de dados
    run_script("load_csv.py")
    
    # 2. Processar dados
    run_script("data_processing.py")
    
    # 3. Rodar o dashboard (Streamlit)
    logger.info("🌐 Iniciando o dashboard em uma nova aba...")
    subprocess.run(["streamlit", "run", "./scr/dashboard.py"])

if __name__ == "__main__":
    main()
