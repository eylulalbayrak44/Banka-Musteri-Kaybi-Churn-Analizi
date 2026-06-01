# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


df = pd.read_csv('Bank Customer Churn Prediction.csv')

print("Veri başarıyla yüklendi!")
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Veri setindeki sütun isimlerini temizle
df.columns = df.columns.str.strip()

# TARGET (Y): Tahmin etmek istediğimiz sütun
target_col = 'churn'

# FEATURES (X): Modelin kullanacağı sütunlar

cols_to_drop = [target_col, 'customer_id', 'RowNumber', 'CustomerId', 'Surname']
actual_drop_cols = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=actual_drop_cols)
y = df[target_col]

# Kategorik ve sayısal sütunları otomatik ayır
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)

print("✅ Sızıntı (Leakage) giderildi. Veri başarıyla ayrıldı.")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Modelleri tanımlıyoruz
models = {
    'Lojistik Regresyon': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Modelleri döngü ile eğitip sonuçları ekrana basıyoruz
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test_trans)
    print(f"\n--- {name} Performans Raporu ---")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

import shap

# En iyi sonuç veren XGBoost modelini analiz ediyoruz
explainer = shap.TreeExplainer(models['XGBoost'])
shap_values = explainer(X_test_trans)

# Grafik çizdirme (Feature Importance)
shap.summary_plot(shap_values, X_test_trans, feature_names=preprocessor.get_feature_names_out())

from statsmodels.stats.contingency_tables import mcnemar

# Tahminleri alıyoruz
p1 = models['Random Forest'].predict(X_test_trans)
p2 = models['XGBoost'].predict(X_test_trans)

# Kontenjans tablosu hesaplama
c1 = np.sum((p1 == y_test) & (p2 != y_test))
c2 = np.sum((p2 == y_test) & (p1 != y_test))

# Testi çalıştır
result = mcnemar([[0, c1], [c2, 0]], exact=True)
print(f"McNemar Testi Sonucu (p-değeri): {result.pvalue:.5f}")

if result.pvalue < 0.05:
    print("Sonuç: Modeller arasındaki fark istatistiksel olarak ANLAMLIDIR.")
else:
    print("Sonuç: Fark istatistiksel olarak anlamlı değildir.")

import joblib

# En iyi modelimizi (XGBoost) ve veri ön işleyicimizi kaydediyoruz
joblib.dump(models['XGBoost'], 'bank_churn_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ Model ve Preprocessor başarıyla kaydedildi! (MLOps Adımı)")

# Modelin yanlış tahmin ettiği (Hata yaptığı) satırları bulalım
y_pred = models['XGBoost'].predict(X_test_trans)
errors = X_test[y_test != y_pred].copy()
errors['True_Status'] = y_test[y_test != y_pred]
errors['Predicted_Status'] = y_pred[y_test != y_pred]

print(f"Toplam Hata Sayısı: {len(errors)}")
print("\nModelin en çok yanıldığı ilk 5 müşteri profili:")
errors.head()

from sklearn.model_selection import GridSearchCV

# XGBoost için küçük bir parametre araması yapalım
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=3, scoring='f1')
grid.fit(X_train_res, y_train_res)

print("✅ En iyi parametreler bulundu:", grid.best_params_)

import joblib

# En iyi modelini ve veri işleyiciyi kaydet
joblib.dump(models['XGBoost'], 'xgb_churn_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ Model ve Preprocessor 'xgb_churn_model.pkl' olarak kaydedildi.")