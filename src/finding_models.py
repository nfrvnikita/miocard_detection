"""Импорт библиотек"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score


COLORS = ['#D94747', '#26AF4D']


def model(classifier,
          X_train=pd.DataFrame,
          X_valid=pd.DataFrame,
          y_train=pd.DataFrame,
          y_valid=pd.DataFrame):
    """Функция для определения скора.

    Args:
        classifier: класс модели.
        X_train (pd.DataFrame): тренировочный датасет.
        X_valid (pd.DataFrame): валидационный датасет.
        y_train (pd.DataFrame): тренировочный таргет.
        y_valid (pd.DataFrame): валидационный таргет.
    """
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_valid)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    print(f'Accuracy: {accuracy_score(y_valid, prediction)}')
    print(f'CV: {cross_val_score(classifier, X_train, y_train, cv=cv, scoring="roc_auc").mean()}')
    print(f'ROC_AUC Score: {roc_auc_score(y_valid, prediction)}')


def model_evaluation(classifier,
                     X_valid=pd.DataFrame,
                     y_valid=pd.DataFrame):
    """Функция для определения скора.

    Args:
        classifier: класс модели.
        X_valid (pd.DataFrame): валидационный датасет.
        y_valid (pd.DataFrame): валидационный таргет.
    """
    conf_m = confusion_matrix(y_valid, classifier.predict(X_valid))
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    counts = [value for value in conf_m.flatten()]

    percentages = [f'{0:.2%}'.format(value) for value in
                   conf_m.flatten()/np.sum(conf_m)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts,
                                                        percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(conf_m, annot=labels, cmap=COLORS, fmt='')

    print(classification_report(y_valid, classifier.predict(X_valid)))
