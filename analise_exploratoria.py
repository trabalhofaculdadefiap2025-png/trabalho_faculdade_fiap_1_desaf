import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\333025\Documents\tech_01\database\data.csv")
print(df.head())
print(f"Informações sobre as colunas (dtype):",df.info())
print(f"Número de linhas e colunas:",df.shape)


#Balanciamento de classes
diagnosis_counts = df['diagnosis'].value_counts()
print(f"Contagem de diagnósticos (M = Maligno, B = Benigno):",diagnosis_counts)

total_samples = len(df)
print(f"\nPorcentagem de cada diagnóstico:",diagnosis_counts / total_samples * 100)

#entenddo escala
print(df.describe())

#Visualizações
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# ---  Correlação Geral ---
df_geral = df.copy()
df_geral['diagnosis'] = df_geral['diagnosis'].map({'M': 1, 'B': 0})
df_geral.drop('id', axis=1, inplace=True)

corr_matrix_geral = df_geral.corr()
mask_geral = np.triu(np.ones_like(corr_matrix_geral, dtype=bool))

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix_geral, mask=mask_geral, annot=True, fmt=".1f", cmap='coolwarm',
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={'size': 8})
plt.title('Mapa de Calor de Correlação Geral', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Análise Bivariada (Variável vs. Target)
features_to_plot = [
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "radius_worst",
    "perimeter_worst",
    "area_worst",
    "area_se",
    "texture_worst",
]

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y=feature, data=df_geral, palette='Set2')
    plt.title(f'Distribuição de {feature} por Diagnóstico', fontsize=16)
    plt.xlabel('Diagnóstico (B = Benigno, M = Maligno)', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.show()