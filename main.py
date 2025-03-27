import os
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, folder = 'src'):
    logger.info(f"🔹 Executando {script_name}...")
    start_time = time.time()
    try:
        # Verifica se o script está na pasta src ou data
        script_path = os.path.join(BASE_DIR, folder, script_name)
        subprocess.run(["python", script_path], check=True)
        logger.info(f"✅ {script_name} executado com sucesso em {time.time() - start_time:.2f} segundos.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao executar {script_name}: {e}")
    except FileNotFoundError:
        logger.error(f"❌ Arquivo {script_name} não encontrado no caminho {script_path}")

def main():
    logger.info("🚀 Iniciando Pipeline de Vendas de Video Games...")
    
    # 1. Carregar dados para o banco de dados
    run_script("load_csv.py", folder = 'data')
    
    # 2. Processar dados
    run_script("data_processing.py")
    
    # 3. Rodar o dashboard (Streamlit)
    logger.info("🌐 Iniciando o dashboard em uma nova aba...")
    dashboard_path = os.path.join(BASE_DIR, 'src', 'dashboard.py')
    subprocess.run(["streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()
