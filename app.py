"""Импорт библиотек"""
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from src.preproccesing import dummy_encode


DATA_PATH = '/home/nfrvnikita/miocard/data/heart.csv'
MODEL_PATH = '/home/nfrvnikita/miocard/models/catboost_model'

st.set_page_config(layout='wide',
                   page_title='Прогнозирование ССЗ',
                   page_icon='🏥',
                   initial_sidebar_state="expanded")
st.write('# Инструмент предсказания сердечно-сосудистых заболеваний')
st.sidebar.title('Введите данные:')

with st.expander("Приложение Streamlit для построения модели прогнозирования ССЗ", expanded=False):
    st.write('- Сердечно-сосудистые заболевания (ССЗ) занимают первое место среди причин смерти во \
        всем мире, ежегодно унося жизни 17,9 млн. человек, что составляет 31% всех смертей в мире.')
    st.write('- Люди с сердечно-сосудистыми заболеваниями или с высоким сердечно-сосудистым риском \
        (из-за наличия одного или нескольких факторов риска, таких как гипертония, диабет, \
        гиперлипидемия или уже установленное заболевание) нуждаются в раннем выявлении и лечении, \
        и в этом случае модель машинного обучения может оказать большую помощь.')


def user_input_features() -> pd.DataFrame:
    """Функция, которая помогает пользователю выбрать характеристики.

    Returns:
        pd.DataFrame: данные, введенные пользователем.
    """
    age = st.sidebar.slider('Возраст', 0, 130, 25)
    sex = st.sidebar.selectbox('Пол', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('Тип боли в груди', ('TA', 'ATA', 'NAP', 'ASY'))
    resting_bp = st.sidebar.slider('Артериальное давление в покое', 0, 200, 25)
    cholesterol = st.sidebar.slider('Холестерин', 0, 400, 25)
    fasting_bs = st.sidebar.select_slider('Диабет', ('N', 'Y'))
    resting_ecg = st.sidebar.selectbox('ЭКГ в покое', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('Максимальное сердцебиение', 0, 250, 25)
    exercise_angina = st.sidebar.slider('Стенокардия, вызванная нагрузкой', 0.1, 2.5, 0.2)
    oldpeak = st.sidebar.slider('Ишемическая депрессия сегмента ST', 0.1, 10.0, 2.5)
    st_slope = st.sidebar.selectbox('Наклон пикового сегмента ST', ('Up', 'Flat', 'Down'))

    features = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    return pd.DataFrame(data=features, index=[0])


user_data = user_input_features()

st.subheader('Введённые пользователем параметры')
st.write(user_data)

# Загружаем исходные данные для дальнейшей обработки
raw_data = pd.read_csv(DATA_PATH)
data = raw_data.drop(columns=['HeartDisease'])
concat_data = pd.concat([user_data, data], axis=0)

# Обрабатываем входные данные
list_columns = list(concat_data.columns)

num_features = []
cat_features = []

for col in list_columns:
    if len(concat_data[col].unique()) > 6:
        num_features.append(col)
    else:
        cat_features.append(col)

final_data = dummy_encode(concat_data, cat_features)
final_data[num_features] = StandardScaler().fit_transform(final_data[num_features])
user_final_data = final_data[:1]

# Загружаем веса предобученной модели
catboost = CatBoostClassifier()
loaded_model = catboost.load_model(MODEL_PATH)

# Применяем модель для предсказания таргета
prediction = loaded_model.predict(user_final_data)
prediction_proba = loaded_model.predict_proba(user_final_data)


st.subheader('Прогноз')
heart_disease = np.array([0, 1])
st.write(heart_disease[prediction])

st.subheader('Вероятность прогноза')
st.write(prediction_proba)
