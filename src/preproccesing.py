"""Импорт библиотек"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def dummy_encode(data=pd.DataFrame, columns=list) -> pd.DataFrame:
    """Функция для кодирования категориальных признаков.

    Args:
        data (pd.DataFrame): наш датасет.
        columns (list): признаки, которые необходимо категоризовать.

    Returns:
        pd.DataFrame: обработанный датасет.
    """
    encoded_df = pd.get_dummies(data,
                                columns=columns,
                                drop_first=True,
                                dtype=int)
    return encoded_df


def standard_norm(X_train=pd.DataFrame, X_valid=pd.DataFrame,
                  columns=list) -> pd.DataFrame:
    """Функция для кодирования числовых признаков.

    Args:
        X_train (pd.DataFrame): тренировочный набор данных.
        X_valid (pd.DataFrame): валидационный набор данных.
        columns (list): признаки, которые необходимо нормализовать.

    Returns:
        X_train (pd.DataFrame): нормализованный тренировочный датасет.
        X_valid (pd.DataFrame): нормализованный валид. датасет.
    """
    scaler = StandardScaler()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_valid[columns] = scaler.transform(X_valid[columns])
    return X_train, X_valid
