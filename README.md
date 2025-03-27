# Análise de Vendas de Jogos Eletrônicos

Este projeto consiste em uma análise completa de vendas de jogos eletrônicos, desde o processamento dos dados até a criação de um dashboard interativo e previsões futuras.

## Estrutura do Projeto

```
VideoGameSales/
├── data/
│   └── vgsales.csv         # Dados brutos de vendas de jogos
├── src/
│   ├── data_processing.py  # Script para processamento dos dados
│   ├── database.py         # Configuração e operações com PostgreSQL
│   ├── dashboard.py        # Aplicação Streamlit
├── requirements.txt        # Dependências do projeto
└── README.md              # Este arquivo
```

## Funcionalidades

### 1. Processamento de Dados
- Leitura e limpeza do arquivo CSV de vendas de jogos
- Tratamento de valores ausentes e outliers
- Normalização e formatação dos dados

### 2. Banco de Dados
- Armazenamento dos dados processados em PostgreSQL
- Estrutura otimizada para consultas rápidas
- Backup e recuperação de dados

### 3. Dashboard Interativo
Visualizações interativas usando Streamlit:
- Top jogos mais vendidos por país
- Análise por gênero
- Ranking mundial
- Vendas por região
- Evolução temporal por ano

### 4. Previsão de Vendas (Futuro)
- Modelo de machine learning para previsão de vendas
- Análise de tendências históricas
- Projeções futuras por gênero e região

## Requisitos
- Python 3.8+
- PostgreSQL
- Bibliotecas Python (ver requirements.txt)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/raphaeleng-94/VideoGameSales.git
cd VideoGameSales
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o banco de dados PostgreSQL:
- Crie um banco de dados
- Configure as credenciais no .env

## Uso

1. Processamento dos dados:
```bash
python src/data_processing.py
```

2. Executar o dashboard:
```bash
streamlit run src/dashboard.py
```

## Contribuição
Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## Licença
Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
