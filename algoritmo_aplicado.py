import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Importa o Random Forest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import RandomOverSampler


df = pd.read_csv(r"C:\Users\333025\Documents\tech_01\database\data.csv")

features_to_use = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst',
    'perimeter_worst', 'area_worst', 'area_se', 'texture_worst',
    'diagnosis'
]

df_selected = df[features_to_use]

if 'id' in df_selected.columns:
    df_selected.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df_selected.columns:
    df_selected.drop('Unnamed: 32', axis=1, inplace=True)

X = df_selected.drop('diagnosis', axis=1)
y = df_selected['diagnosis']
y = y.map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'minmax_scaler.joblib')


# =========================================================
# --- SEÇÃO 1: DECISION TREE CLASSIFIER (Modelo de Deploy) ---
# =========================================================
print("\n" + "="*50)
print("             DECISION TREE CLASSIFIER")
print("="*50)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_scaled, y_train_resampled)

# Salvar o modelo treinado
joblib.dump(dtree, 'decision_tree_model.joblib')

y_pred_dtree = dtree.predict(X_test_scaled)

# 1. Feature Importance
feature_importances = pd.Series(dtree.feature_importances_, index=X.columns)

print("\n--- Feature Importance (Decision Tree) ---")
print(feature_importances.sort_values(ascending=False))

# 2. Relatório de Classificação
print("\n--- Relatório de Classificação (Decision Tree) ---")
print(classification_report(y_test, y_pred_dtree, target_names=['Benigno (0)', 'Maligno (1)']))

# Matriz de Confusão
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
disp_dtree = ConfusionMatrixDisplay(confusion_matrix=cm_dtree, display_labels=['Benigno', 'Maligno'])
disp_dtree.plot(cmap=plt.cm.Greens)
plt.title('Matriz de Confusão: Decision Tree')
plt.show()


# =========================================================
# --- SEÇÃO 2: RANDOM FOREST CLASSIFIER (Modelo de Comparação) ---
# =========================================================
print("\n" + "="*50)
print(" RANDOM FOREST CLASSIFIER")
print("="*50)

rforest = RandomForestClassifier(random_state=42, n_estimators=100)
rforest.fit(X_train_scaled, y_train_resampled)

y_pred_rforest = rforest.predict(X_test_scaled)

print("\n--- Relatório de Classificação (Random Forest) ---")
print(classification_report(y_test, y_pred_rforest, target_names=['Benigno (0)', 'Maligno (1)']))

cm_rforest = confusion_matrix(y_test, y_pred_rforest)
disp_rforest = ConfusionMatrixDisplay(confusion_matrix=cm_rforest, display_labels=['Benigno', 'Maligno'])
disp_rforest.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão: Random Forest')
plt.show()