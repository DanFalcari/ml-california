import pandas as pd


def dataframe_coeficientes(coefs, colunas):
    """
    Cria um DataFrame organizado com os coeficientes de um modelo linear.

    Parameters
    ----------
    coefs : array-like of shape (n_features,)
        Array contendo os coeficientes do modelo linear. Pode ser obtido através do 
        atributo `.coef_` dos modelos scikit-learn.
        
    colunas : array-like of shape (n_features,)
        Lista com os nomes das features correspondentes aos coeficientes. Deve estar na
        mesma ordem usada no treinamento do modelo.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo:
        - Index: Nomes das features
        - Coluna 'coeficiente': Valores dos coeficientes ordenados por magnitude 
    """
    return pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"]).sort_values(
        by="coeficiente"
    )
