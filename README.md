# 🏠 California Housing Price Prediction

## 📌 Visão Geral

Este projeto é uma aplicação de machine learning com interface interativa via Streamlit, capaz de prever o preço médio de imóveis na Califórnia com base em atributos demográficos, socioeconômicos e geográficos.

## 🎯 Objetivo

Desenvolver um modelo preditivo de preços de imóveis com visualização geográfica e aplicação interativa que permite ao usuário:

### Funcionalidades da Aplicação

* Escolher um condado da Califórnia
* Inserir atributos de um imóvel
* Informar renda anual (em U\$\$)
* Obter uma previsão do valor médio do imóvel (em US\$)

## 🛠️ Stack Tecnológico

* **Python**
* **Pandas / NumPy** – Manipulação e análise de dados
* **Matplotlib / Seaborn** – Visualização de dados
* **GeoPandas / Shapely** – Análise de dados geoespaciais
* **Folium / Pydeck** – Mapas interativos
* **Scikit-learn** – Treinamento e avaliação de modelos de regressão
* **Streamlit** – Interface web interativa
* **Joblib** – Persistência do modelo
* **Parquet** – Armazenamento eficiente de dados

### ⚙️ Ferramentas Utilizadas

* JupyterLab (experimentação e testes)
* Git (controle de versão)
* Conda (ambiente virtual)
* VS Code (editor principal)

## 🧩 Principais Features Técnicas

* **Pipeline de Machine Learning** com `ColumnTransformer`, `PolynomialFeatures`, `StandardScaler` e `GridSearchCV`
* **Engenharia de Atributos (Feature Engineering)** com variáveis derivadas como:

  * `rooms_per_household`
  * `bedrooms_per_room`
  * `population_per_household`
  * `median_income_cat` (renda categorizada)
* **Mapas Geoespaciais** com Folium (EDA) e Pydeck (visualização interativa)
* **Cache de performance** com `@st.cache_data` e `@st.cache_resource` no Streamlit

## 🧪 Metodologia

O projeto foi desenvolvido com uma estrutura modular, orientada à experimentação e replicabilidade. As principais etapas foram:

### Análise Exploratória de Dados (EDA)

* Estudo das distribuições, outliers e correlações
* Mapas interativos para compreensão da geolocalização
* Relações visuais entre renda, localização e preço dos imóveis

### Limpeza e Preparação dos Dados

* Tratamento de valores ausentes e inconsistentes
* Conversão de tipos de dados e padronização
* Balanceamento de categorias para evitar viés

### Engenharia de Atributos (Feature Engineering)

* Novos atributos criados a partir dos dados originais
* Categorização de variáveis contínuas (e.g. `median_income_cat`)
* Seleção dos atributos com maior correlação com a variável-alvo

### Estrutura Modular de Treinamento de Modelos

* Uso de `Pipeline` com transformações encadeadas
* Testes com múltiplos modelos:

  * `DummyRegressor`, `LinearRegression`, `ElasticNet` e `Ridge`
* Otimização com `GridSearchCV` e validação cruzada
* Avaliação com múltiplas métricas:

  * **R²**, **RMSE** (Root Mean Squared Error), **MAE** (Erro Absoluto Médio)

### Persistência e Deploy

* Salvamento do modelo final com `joblib`
* Armazenamento de dados em `.parquet`
* Deploy da aplicação com Streamlit, com carregamento otimizado por cache

## 📊 Resultados

Durante os testes, diversos modelos foram treinados e avaliados com validação cruzada. Abaixo estão os principais resultados obtidos:

| Modelo                            | R² (test)  | MAE (test) | RMSE (test) | Tempo de Treinamento |
| --------------------------------- | ---------- | ---------- | ----------- | -------------------- |
| DummyRegressor                    | -0.0004    | 76.702     | 95.975      | 0.035s               |
| Linear Regression                 | 0.6681     | 40.780     | 55.277      | 0.151s               |
| Linear Regression (Target Scaled) | 0.6783     | 38.676     | 54.414      | 0.196s               |
| ElasticNet (Grid Search)          | 0.7139     | 36.003     | 51.309      | 57.505s              |
| Ridge (Grid Search)               | **0.7234** | **35.363** | **50.444**  | 0.754s               |

### 🧠 Principais Insights e Melhorias

* O modelo baseline (`DummyRegressor`) reforçou a necessidade de modelagem avançada.
* O uso de **engenharia de atributos** e **normalização da variável alvo** trouxe ganhos consistentes em performance.
* A aplicação de **GridSearchCV** nos modelos ElasticNet e Ridge permitiu ajustes finos nos hiperparâmetros, elevando o desempenho.
* O modelo final (`Ridge`) apresentou uma melhoria de **mais de 33% na RMSE** comparado à regressão linear simples.
* A escalabilidade da arquitetura permite testar novos modelos e ajustes com mínimos no código.

### 🔗 Acesse a Aplicação

A aplicação pode ser acessada via Streamlit Cloud:

👉 [![Open in Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://danfalcari-ml-california-home-iilyln.streamlit.app/)

## Organização do projeto

```
├── .gitignore             <- Arquivos e diretórios a serem ignorados pelo Git
├── requirements.txt       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE                <- MIT License
├── README.md              <- README principal para desenvolvedores que usam este projeto.
├── home.py                <- Código da aplicação web 
|
├── dados                  <- Arquivos de dados para o projeto.
|
├── modelos                <- Modelos de machine learning
|
├── notebooks              <- Cadernos Jupyter. A convenção de nomenclatura é um número (para ordenação),
│                           as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                           `01-fb-exploracao-inicial-de-dados`.
│
|   └──src                 <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py     <- Torna um módulo Python
|      ├── auxiliares.py   <- 
|      ├── config.py       <- Configurações básicas do projeto
|      ├── graficos.py     <- Scripts para criar visualizações exploratórias e orientadas a resultados
|      └── models.py       <-
|
├── referencias            <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── relatorios             <- Análises geradas
│   └── imagens            <- Gráficos e figuras gerados
```


## Como Reproduzir o Ambiente

Para facilitar a replicação do projeto, todas as dependências foram listadas no arquivo requirements.txt, o que permite a instalação com pip, garantindo compatibilidade ampla, inclusive com o Streamlit Cloud.

1. Faça o clone do repositório que será criado a partir deste modelo.

    ```bash
    git git@github.com:DanFalcari/ml-california.git
    ```

2. Crie um ambiente virtual para o seu projeto utilizando o gerenciador de ambientes de sua preferência.

    a. Caso esteja utilizando o `conda`, exporte as dependências do ambiente para o arquivo `ambiente.yml`:

      ```bash
      conda env export > ambiente.yml
      ```

    b. Caso esteja utilizando outro gerenciador de ambientes, use o para o arquivo `requirements.txt`
    Adicione o arquivo ao controle de versão, removendo o arquivo `ambiente.yml`.


