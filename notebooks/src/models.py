import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def construir_pipeline_modelo_regressao(regressor, preprocessor=None, target_transformer=None):
   """
    Constrói um pipeline completo para modelagem de regressão, com opções de pré-processamento
    e transformação do target.

    Parameters
    ----------
    regressor : sklearn estimator
        Modelo de regressão a ser utilizado (ex: LinearRegression).
        
    preprocessor : sklearn transformer, optional
        Pipeline de pré-processamento para as features (ex: ColumnTransformer).
        Se None, nenhum pré-processamento é aplicado.
        
    target_transformer : sklearn transformer, optional
        Transformador para a variável target (ex: PowerTransformer).
        Se None, o target não é transformado.

    Returns
    -------
    sklearn.pipeline.Pipeline ou sklearn.compose.TransformedTargetRegressor
        Pipeline completo pronto para treinamento.
    """      
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model


def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """
    Executa validação cruzada para um modelo de regressão, retornando métricas de avaliação.

    Parameters
    ----------
    X : array-like ou DataFrame de shape (n_samples, n_features)
        Dados de entrada.
        
    y : array-like de shape (n_samples,)
        Valores target.
        
    regressor : sklearn estimator
        Modelo de regressão a ser avaliado.
        
    preprocessor : sklearn transformer, optional
        Pipeline de pré-processamento para as features.
        
    target_transformer : sklearn transformer, optional
        Transformador para a variável target.
        
    n_splits : int, default=5
        Número de folds para validação cruzada.
        
    random_state : int, default=RANDOM_STATE
        Seed para reprodutibilidade do KFold.

    Returns
    -------
    dict
        Dicionário com arrays contendo:
        - fit_time: Tempos de treino para cada fold
        - score_time: Tempos de predição para cada fold
        - test_r2: R² scores para cada fold
        - test_neg_mean_absolute_error: MAE scores para cada fold
        - test_neg_root_mean_squared_error: RMSE scores para cada fold
    """    
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organiza_resultados(resultados):

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
