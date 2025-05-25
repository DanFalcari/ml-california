# Bibliotecas 
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st
from joblib import load

# Caminhos para os dados e modelo
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

# Função para carregar dados já tratados
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

# Função para carregar e preparar os dados geográficos
@st.cache_data
def carregar_dados_geo():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Corrige geometrias inválidas e orienta os polígonos
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if isinstance(geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extrai coordenadas dos polígonos para visualização no mapa
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)
    return gdf_geo

# Carrega modelo de machine learning treinado
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

# --- Inicialização ---

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# Título do app
st.title("🏠 Previsão de Preços de Imóveis na Califórnia")

# Lista de condados disponíveis
condados = sorted(gdf_geo["name"].unique())

# Interface dividida em duas colunas
coluna1, coluna2 = st.columns(2)

# --- Coluna 1: Formulário de entrada de dados ---

with coluna1:
    with st.form(key="formulario"):
        selecionar_condado = st.selectbox("Selecione um Condado", condados)

        # Extrai os dados médios do condado selecionado
        dados_condado = gdf_geo.query("name == @selecionar_condado")

        longitude = dados_condado["longitude"].values
        latitude = dados_condado["latitude"].values

        housing_median_age = st.number_input("Idade do imóvel (anos)", min_value=1, max_value=50, value=10)
        median_income = st.slider("Renda média (milhares de US$)", 5.0, 100.0, 45.0, 5.0)
        median_income_scale = median_income / 10  # Escala usada no treinamento

        # Variáveis derivadas do dataset geográfico
        total_rooms = dados_condado["total_rooms"].values
        total_bedrooms = dados_condado["total_bedrooms"].values
        population = dados_condado["population"].values
        households = dados_condado["households"].values
        ocean_proximity = dados_condado["ocean_proximity"].values

        # Categorização da renda
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income_scale, bins=bins_income)

        rooms_per_household = dados_condado["rooms_per_household"].values
        bedrooms_per_room = dados_condado["bedrooms_per_room"].values
        population_per_household = dados_condado["population_per_household"].values

        # Monta DataFrame de entrada para o modelo
        entrada_modelo = pd.DataFrame({
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income_scale,
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        })

        # Botão para executar a previsão
        botao_previsao = st.form_submit_button("🔍 Prever Preço")

    # Exibe o resultado da previsão
    if botao_previsao:
        preco = modelo.predict(entrada_modelo)
        st.metric(label="💰 Preço previsto (US$)", value=f"{preco[0][0]:,.2f}")

# --- Coluna 2: Mapa interativo ---

with coluna2:
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    # Camada com todos os condados
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],  # Azul claro
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )

    # Destaque do condado selecionado
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=dados_condado[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 100],  # Vermelho
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    # Tooltip interativo
    tooltip = {
        "html": "<b>Condado:</b> {name}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"},
    }

    # Criação do mapa
    mapa = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(mapa)
