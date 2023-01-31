# 1.1. Gostaria de chegar de manhã na minha mesa e ter um lugar único onde eu possa observar o portfólio da House
# Rocket. Nesse portfólio, eu tenho interesse:
# 1. Filtros dos imóveis por um ou várias regiões .
# 2. Escolher uma ou mais variáveis para visualizar.
#3. Observar o número total de imóveis, a média de preço, a média da sala de estar e também a média do preço por metro
# quadrado em cada um dos códigos postais.
# 4. Analisar cada uma das colunas de um modo mais descrito.
# 5. Uma mapa com a densidade de portfólio por região e também densidade de preço.
# 6. Checar a variação anual de preço.
# 7. Checar a variação diária de preço.
# 8. Conferir a distribuição dos imóveis por:
# - preço, - Número de quartos - Numero de banheiros - Numero de  andares - Vista para a água ou não

import pandas as pd
import streamlit as st
import numpy as np
import folium #biblioteca para mapas
import geopandas

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster #adicionar pontos mapa

import plotly.express as px
from datetime import datetime

st.set_page_config(layout = 'wide') #deixa a tabela mais ampla na tela

@st.cache(allow_output_mutation = True)

def get_data(path): #pode fazer do jeito tradicional mas assim eh mais rapido
    data = pd.read_csv(path)

    return data

@st.cache(allow_output_mutation = True) #para fazer soh uma vez
def get_geofile(url):
    geofile = geopandas.read_file(url) #passa uma url e ele le o json

    return geofile

def set_feature(data):
    # add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview(data):
    # Data Overview
    f_attributes = st.sidebar.multiselect('Enter columns ', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode ', data['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe(data)

    c1, c2 = st.columns((1, 1))  # colocar graficos lado a lado, numero indica a largura
    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')  # on indica a coluna q vai unir eles
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')  # merge soh da pra fazer 2 em 2
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    # Diferenca do merge para o cancat, merge garante a uniao de item a item

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']  # Renomear as colunas

    c1.header('Average Values')
    # st.dataframe(df, width = 800, height = 600)
    c1.dataframe(df, width=800, height=600)

    # st.write(df.head())
    # Statistic Descritive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])

    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df5 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df5.columns = ['ATTRIBUTES', 'MAXIMO', 'MINIMO', 'MEDIA', 'MEDIANA', 'DESVIO PADRÃO']

    c2.header('Descriptive Analysis')
    # st.dataframe(df5, height = 600)
    c2.dataframe(df5, height=600)

    return None

def portfolio_density(data,geofile):
    # Densidade de Portfolio
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    df = data.sample(1000)

    # Bae Map - Folium

    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():  # vai pegar a linha e o nome da coluna
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R$ {0} on: {1}. Features: {2} sqft, {3} bedrooms, '
                            '{4} bathrooms, year built: {5}'.format(row['price'],
                                                                    row['date'],
                                                                    row['sqft_living'],
                                                                    row['bedrooms'],
                                                                    row['bathrooms'],
                                                                    row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # df.sample(10)

    geofile = geofile[
        geofile['ZIP'].isin(df['ZIP'].tolist())]  # filtra os zip do arquivo pra pegar soh os que tem no data

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  default_zoom_start=15)  # criaçao do mapa base

    folium.Choropleth(data=df,  # plot de densidade por cor
                      geo_data=geofile,
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIP',  # junçao dos dados
                      fill_color='YlOrRd',
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      legend_name='AVG PRICE').add_to(region_price_map)

    with c2:
        folium_static(region_price_map)

    return None

def  commercial_distribution(data):
    # Distribuição dos imoveis por categorias comerciais

    st.sidebar.title('Commertial Options')
    st.title('Commertial Attributes')

    # Average Price per year

    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    st.header('Average Price per yaer built')

    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')

    st.plotly_chart(fig, use_container_width=True)

    # Average Price per date
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    st.header('Average Price per day')
    st.sidebar.subheader('Select Max Date')

    # filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # data filtering
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ----- Histograma
    st.header('Price Dsitribution')
    st.sidebar.subheader('Select Max Price')

    # filter
    max_price = int(data['price'].max())
    min_price = int(data['price'].min())
    avg_price = int(data['price'].mean())

    # data filtering
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)  # nbins numero de barras
    st.plotly_chart(fig, use_container_width=True)

    return None

def  attributes_distribution(data):
    # ===================
    # Distribuicao dos imoveis por categorias fisicas
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filters

    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filters

    f_floors = st.sidebar.selectbox('Max number of floors', sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with Water View')

    c1, c2 = st.columns(2)

    # House per floors
    c1.header('Houses per floor')
    df = data[data['floors'] <= f_floors]

    # plot
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per water view
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == '__main__':
    #ETL
    #data extraction
    path = 'kc_house_data.csv'
    url = 'https://gisdata.seattle.gov/server/rest/services/COS/Seattle_City_Limits/MapServer/2/query?outFields=*&where=1%3D1&f=geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #transsformation
    data = set_feature(data)

    overview(data)

    portfolio_density(data,geofile)

    commercial_distribution(data)

    attributes_distribution(data)

    #loading carregar dados em uma API ou banco de dados
    #get data


