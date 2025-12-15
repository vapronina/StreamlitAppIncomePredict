import streamlit as st
import pandas as pd
import pickle

# настройка страницы

st.set_page_config(
    page_title="Income Prediction",
    page_icon="💸",
    layout="centered")
# фон
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1b2e 0%, #2d2346 100%);
        color: #f5f5f5;
    }
    h1, h2, h3, p, label {
        color: #f5f5f5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

# загрузка модели

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# заголовки
st.title("Income Prediction App 💸")
st.subheader("Узнай, превысит ли твой доход 50k")
st.write("Заполни данные ниже и получи прогноз от модели")
st.divider()

# ввод данных
age = st.slider("Возраст", 18, 90, 50)
education_num = st.slider("Уровень образования (education-num)", 1, 16, 15)

hours_per_week = st.number_input(
    "Часов работы в неделю",
    min_value=1,
    max_value=100,
    value=50)

sex = st.selectbox("Пол", ["Male", "Female"])

marital_status = st.selectbox(
    "Семейное положение",
    ["Never-married", "Married", "Divorced"])

workclass = st.selectbox(
    "Тип занятости",
    ["Private", "Self-emp", "State-gov", "Federal-gov"])

occupation = st.selectbox(
    "Род занятий",
    [
        "Exec-managerial",
        "Prof-specialty",
        "Sales",
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical"])
st.divider()

# формирование входа
X_input = pd.DataFrame(columns=columns)
X_input.loc[0] = 0

# числовые признаки
X_input.loc[0, "age"] = age
X_input.loc[0, "education-num"] = education_num
X_input.loc[0, "hours-per-week"] = hours_per_week

# числовые признаки, которые пользователь не вводит
for col in ["capital-gain", "capital-loss", "fnlwgt"]:
    if col in X_input.columns:
        X_input.loc[0, col] = 0

# категориальные признаки
if "sex_" + sex in X_input.columns:
    X_input.loc[0, "sex_" + sex] = 1

if "marital-status_" + marital_status in X_input.columns:
    X_input.loc[0, "marital-status_" + marital_status] = 1

if "workclass_" + workclass in X_input.columns:
    X_input.loc[0, "workclass_" + workclass] = 1

if "occupation_" + occupation in X_input.columns:
    X_input.loc[0, "occupation_" + occupation] = 1

# наконец-то предсказание
if st.button("Предсказать доход 🔮"):
    # берем числовые признаки в правильном порядке иначе ошибку выдает
    num_cols = list(scaler.feature_names_in_)
    X_scaled = X_input.copy()
    X_scaled[num_cols] = scaler.transform(X_scaled[num_cols])
    proba = model.predict_proba(X_scaled)[0, 1]
    st.metric(
        label="Вероятность дохода > 50k",
        value=f"{proba:.2f}")
    if proba > 0.5:
        st.success("Доход превысит 50k 💰")
        st.balloons()
    else:
        st.warning("Доход не превысит 50k")
