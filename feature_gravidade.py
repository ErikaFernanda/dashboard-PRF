import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pickle


folder_path = "./dataset/"
anos = range(2016, 2020)
df_total = []
for ano in anos:
    file_path = f"{folder_path}/datatran{ano}.csv"
    df = pd.read_csv(file_path, sep=";", encoding="latin-1")
    df["hour"] = pd.to_datetime(df["horario"], errors="coerce").dt.hour
    df["morto"] = pd.to_numeric(df["mortos"], errors="coerce").fillna(0)
    df["feridos_graves"] = pd.to_numeric(df["feridos_graves"], errors="coerce").fillna(0)
    df["grave"] = ((df["morto"] > 0) | (df["feridos_graves"] > 0)).astype(int)
    df_total.append(df)
df_all = pd.concat(df_total).dropna(subset=["hour", "dia_semana", "condicao_metereologica"])

features = ["hour", "dia_semana", "condicao_metereologica"]
X = df_all[features]
y = df_all["grave"]


df_balanced = pd.concat([
    df_all[df_all["grave"] == 1],
    df_all[df_all["grave"] == 0].sample(n=df_all["grave"].sum(), random_state=42)
])

X_bal = df_balanced[features]
y_bal = df_balanced["grave"]


categorical_cols = ["dia_semana", "condicao_metereologica"]
numerical_cols = ["hour"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, stratify=y_bal, test_size=0.2, random_state=42
)


param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Melhores parâmetros:", grid_search.best_params_)


y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

with open("modelo_rf_undersampling.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)

print("Modelo salvo como modelo_rf_undersampling.pkl")
