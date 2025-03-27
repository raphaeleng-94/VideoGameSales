import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from dotenv import load_dotenv

load_dotenv()

# Configurações do banco de dados
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('POSTGRES_PORT')
DB_NAME = os.getenv('POSTGRES_DB')

# Conectar ao banco de dados
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Lendo os dados do banco de dados
query = "SELECT * FROM vgsales"
df = pd.read_sql(query, engine)

# Configurar o Streamlit
st.set_page_config(page_title="Dashboard de Vendas", layout="wide")

st.title('📊 Dashboard de Vendas de Video Games')

# =========================
# 🔹 SELEÇÃO DE FILTROS
# =========================
st.sidebar.header('Filtros')

# Filtro por gênero
generos = df['Genre'].unique()
genero_selecionado = st.sidebar.multiselect('Selecione o gênero:', generos, default=generos)

# Filtro por região
regioes = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
regiao_selecionada = st.sidebar.multiselect('Selecione a região:', regioes, default=regioes)

# Filtrar dados com base na seleção
df_filtrado = df[df['Genre'].isin(genero_selecionado)]

# =========================
# 🔹 TOP 5 JOGOS MAIS VENDIDOS POR REGIÃO
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader('🏆 Top 5 Jogos Mais Vendidos na América')
    top_5_na = df.nlargest(5, 'NA_Sales')[['Name', 'NA_Sales']]
    fig = px.bar(top_5_na, x='Name', y='NA_Sales', text='NA_Sales', color='NA_Sales', color_continuous_scale='Blues')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='Vendas (milhões)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('🌍 Top 5 Jogos Mais Vendidos na Europa')
    top_5_eu = df.nlargest(5, 'EU_Sales')[['Name', 'EU_Sales']]
    fig = px.bar(top_5_eu, x='Name', y='EU_Sales', text='EU_Sales', color='EU_Sales', color_continuous_scale='Greens')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='Vendas (milhões)')
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader('🌍 Top 5 Jogos Mais Vendidos no Japão')
    top_5_jp = df.nlargest(5, 'JP_Sales')[['Name', 'JP_Sales']]
    fig = px.bar(top_5_jp, x='Name', y='JP_Sales', text='JP_Sales', color='JP_Sales', color_continuous_scale='Purples')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='Vendas (milhões)')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader('🌍 Top 5 Jogos Mais Vendidos Globalmente')
    top_5_global = df.nlargest(5, 'Global_Sales')[['Name', 'Global_Sales']]
    fig = px.bar(top_5_global, x='Name', y='Global_Sales', text='Global_Sales', color='Global_Sales', color_continuous_scale='Oranges')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='Vendas (milhões)')
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔹 VENDAS POR REGIÃO
# =========================
st.subheader('📌 Vendas por Região')

sales_by_region = df_filtrado[regiao_selecionada].sum().reset_index()
sales_by_region.columns = ['Região', 'Vendas']

fig = px.pie(sales_by_region, names='Região', values='Vendas', color='Região', hole=0.4)
fig.update_traces(textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔹 VENDAS POR GÊNERO
# =========================
st.subheader('🎮 Vendas por Gênero')

sales_by_genre = df_filtrado.groupby('Genre')['Global_Sales'].sum().reset_index()
fig = px.bar(sales_by_genre, x='Genre', y='Global_Sales', color='Global_Sales', color_continuous_scale='Plasma')
fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
fig.update_layout(xaxis_title='', yaxis_title='Vendas (milhões)')
st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔹 JOGOS MAIS RENTÁVEIS POR MÊS E ANO
# =========================
st.subheader('💰 Total de Vendas por Ano')

df_filtrado['Year'] = pd.to_numeric(df_filtrado['Year'], errors='coerce')
sales_by_year = df_filtrado.groupby('Year')['Global_Sales'].sum().reset_index()

fig = px.line(sales_by_year, x='Year', y='Global_Sales', markers=True, color_discrete_sequence=['#FF6361'])
fig.update_layout(xaxis_title='Ano', yaxis_title='Vendas (milhões)')
st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔹 PREVISÃO DE VENDAS COM XGBOOST
# =========================
st.subheader('📈 Previsão de Vendas Futuras')

# Treinando o modelo de previsão
data = df_filtrado[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].dropna()

X = data[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y = data['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Interface para previsão
year = st.number_input('Ano para previsão', min_value=int(df['Year'].min()), max_value=2030, value=2025)
na_sales = st.number_input('Vendas na América (milhões)', min_value=0.0, value=2.5)
eu_sales = st.number_input('Vendas na Europa (milhões)', min_value=0.0, value=1.8)
jp_sales = st.number_input('Vendas no Japão (milhões)', min_value=0.0, value=0.7)
other_sales = st.number_input('Vendas em outras regiões (milhões)', min_value=0.0, value=0.5)

if st.button('Prever'):
    future_data = pd.DataFrame({
        'Year': [year],
        'NA_Sales': [na_sales],
        'EU_Sales': [eu_sales],
        'JP_Sales': [jp_sales],
        'Other_Sales': [other_sales]
    })

    prediction = model.predict(future_data)
    st.success(f'📊 Previsão de vendas globais para {year}: **{prediction[0]:.2f} milhões de unidades**')

# Avaliação do modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'✔️ **MSE:** {mse:.2f}')  
st.write(f'✔️ **R²:** {r2:.2f}')  

# =========================
# 🔹 RODAPÉ
# =========================
st.markdown("---")
st.write("🚀 **Desenvolvido por Raphael**")





