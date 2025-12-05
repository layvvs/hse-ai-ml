import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


st.title('Домашнее задание №1 часть 2')

@st.cache_resource
def load_model():
    with open('best.pkl', 'rb') as f:
        return pickle.load(f)

model, scaler, config = load_model().values()

st.header('1) Основные графики из EDA')

st.image('corr.png', caption='Корреляция Пирсона и Спирмена', use_container_width=True)
st.image('pairplot.png', caption='Парные зависимости', use_container_width=True)
st.image('processed-data-corr.png', caption='Корреляция обработанных признаков', use_container_width=True)


st.header('2) Инференс')

uploaded_csv = st.file_uploader('Загрузите CSV файл', type=['csv'])

if uploaded_csv is not None:
    data = pd.read_csv(uploaded_csv, index_col=0)
    st.subheader('Первые строки данных')
    st.write(data.head())

    st.subheader('Основная статистика')
    st.write(data.describe())

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    with st.expander('Показать гистограммы по числовым признакам'):
        for col in num_cols:
            fig, ax = plt.subplots()
            ax.hist(data[col].dropna(), bins=30)
            ax.set_title(f'Histogram: {col}')
            st.pyplot(fig)

    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)

    df_preds = pd.DataFrame({'prediction': preds})
    df_final = pd.concat([df_preds, data], axis=1)

    st.subheader('Предсказания')
    st.write(df_final)

    uploaded_csv_targets = st.file_uploader('Загрузите CSV файл с истинными значениями для проверки модели', type=['csv'])

    if uploaded_csv_targets is not None:
        targets = pd.read_csv(uploaded_csv_targets, index_col=0)
        diff = pd.DataFrame({'diff': np.abs(targets.values.flatten() - preds)})
        validation_df = pd.concat([targets, df_preds, diff, data], axis=1)
        
        st.subheader('Валидация предсказаний')
        st.write(validation_df)

    # if uploaded_csv_targets is not None:
    #     targets = pd.read_csv(uploaded_csv_targets, index_col=0)
    #     r2_score_metric = pd.DataFrame({'R2 score': r2_score(targets, df_preds)})
        
    #     validation_df = pd.concat([targets, df_preds, r2_score_metric, data])
    #     st.subheader('Валидация предсказаний')
    #     st.write(df_final)

st.header('3) Визуализация весов модели')

weights = model.coef_
feature_names = ['mileage',
 'max_power',
 'torque',
 'max_torque_rpm',
 'name',
 'fuel_Diesel',
 'fuel_LPG',
 'fuel_Petrol',
 'seller_type_Individual',
 'seller_type_Trustmark Dealer',
 'transmission_Manual',
 'owner_Fourth & Above Owner',
 'owner_Second Owner',
 'owner_Test Drive Car',
 'owner_Third Owner',
 'seats_4',
 'seats_5',
 'seats_6',
 'seats_7',
 'seats_8',
 'seats_9',
 'seats_10',
 'seats_14',
 'year',
 'km_driven',
 'year_squared',
 'km_driven_squared'
]
features = [feature_names[idx] for idx in range(len(weights))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(features, weights)
ax.set_title('Веса модели')
st.pyplot(fig)