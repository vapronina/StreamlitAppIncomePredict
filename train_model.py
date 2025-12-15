import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# 1.Загружаем данные
df_dirty = pd.read_csv("data.adult.csv")
# заменяем ? на пропуски
df_dirty = df_dirty.replace("?", pd.NA)
# удаляем строки с пропусками
df_clean = df_dirty.dropna().copy()

# 2.Целевая переменная
# бинаризация целевой
df_clean[">50K,<=50K"] = df_clean[">50K,<=50K"].replace(
    {">50K": 1, "<=50K": 0})
y = df_clean[">50K,<=50K"].astype(int)

# 3. Признаки
# все признаки без целевой
X = df_clean.drop(columns=[">50K,<=50K"])
# one hot encoding категориальных признаков
X_ohe = pd.get_dummies(X, drop_first=True)
# числовые признаки
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# 4. Масштабирование
scaler = StandardScaler()
X_ohe[num_cols] = scaler.fit_transform(X_ohe[num_cols])

# 5. Обучение модели
model = GradientBoostingClassifier(
    n_estimators=300,
    criterion="squared_error",
    max_features="sqrt",
    random_state=42)
model.fit(X_ohe, y)

# 6. Сохранение файлов
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X_ohe.columns.tolist(), open("columns.pkl", "wb"))

print("Модель обучена и сохранена")
