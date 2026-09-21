#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score)
sns.set_theme(style="whitegrid")

print("Bibliotecas importadas com sucesso!")

#%%
wine = load_wine()

print("Dataset carregado com sucesso!")

#%%
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(df.head())

#%%
#Dataset

print("Número de amostras:", df.shape[0])
print("Quantidade de colunas: ", df.shape[1])

print("Primeiras linhas:")
print(df.head())

#%%

print(df.describe())

#%%

print ("Características Químicas")

for feature in wine.feature_names:
    print("-", feature)

#%%
#Conhecendo a quantidade de classes e suas distribuições

print("Classes disponíveis: ", wine.target_names)

print("Quantidade de amostras por classe:")
print(df['target'].value_counts())

#%%

#Hora dos gráficos

plt.figure(figsize=(8,5))
sns.countplot(x='target', data=df)

plt.title("Quantidade de amostras por classe")
plt.xlabel("Classe")
plt.ylabel("Quantidade de amostras")
plt.show()

#%%

#Gráficos dos dados químicos

plt.figure(figsize=(8,5))

sns.scatterplot(x='alcohol', y='color_intensity', hue='target', data=df)

plt.title("Álcool vs Intensidade de Cor")
plt.xlabel("Teor Alcoólico")
plt.ylabel("Intensidade de Cor")
plt.show()

#%%

# Matriz de Correlação

plt.figure(figsize=(11, 8))
sns.heatmap(
    df.drop(columns = "target").corr(),
    cmap = "coolwarm",
)

plt.title("Matriz de Correlação")
plt.show()

#%%

# Entre Features e Target

x = df.drop(columns = "target") # Características Químicas

y = df['target'] # Classes do vinho

print("Formato de X:", x.shape)
print("Formato de y:", y.shape)

#%%

#Separando os dados em treino e teste

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#%%

#Escolhendo nosso primeiro modelo: Decision Tree Classifier

modelo_tree = DecisionTreeClassifier(random_state=42, max_depth=3)

modelo_tree.fit(x_train, y_train)

print("Modelo treinado com sucesso!")

#%%

# Previsões do modelo

previsions_tree = modelo_tree.predict(x_test)

print("Previsões realizadas com sucesso!")

print(previsions_tree)

#%%
#Acurácia do modelo

accuracy_tree = accuracy_score(y_test, previsions_tree)

print("Acurácia do modelo Decision Tree Classifier: {:.2f}%".format(accuracy_tree * 100))

#%%
# Matriz de Confusão

ConfusionMatrixDisplay.from_predictions(y_test, previsions_tree, display_labels=wine.target_names)

plt.title("Matriz de Confusão - Decision Tree Classifier"
)

plt.show()

#%%

#Descobrindo a profundidade da folha

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = DecisionTreeClassifier(random_state=42, max_depth=profundidade)
    modelo.fit(x_train, y_train)
    treino.append(modelo.score(x_train, y_train))
    teste.append(modelo.score(x_test, y_test))

#%%

# Gráfico de complexidade e desempenho

plt.figure(figsize=(8,5))

plt.plot(profundidades, treino, label='Treino', marker='o')
plt.plot(profundidades, teste, label='Teste', marker='o')

plt.xlabel("Profundidade da Árvore")
plt.ylabel("Acurácia")
plt.title("Complexidade vs Desempenho - Decision Tree Classifier")
plt.legend()
plt.grid()
plt.show()

#%%
# Outro modelo: Random Forest Classifier

modelo_rf = RandomForestClassifier(max_depth = 5, random_state=42, n_estimators=100)
modelo_rf.fit(x_train, y_train)
previsoes_rf = modelo_rf.predict(x_test)

print("Random Forest Treinado hehe")

#%%
# Acurácia do modelo Random Forest

accuracy_rf = accuracy_score(y_test, previsoes_rf)

print("Acurácia do modelo Random Forest Classifier: {:.2f}%".format(accuracy_rf * 100))

#%%
# Modelo KNN (Validação cruzada)

modelo_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5)) 
])

modelo_knn.fit(x_train, y_train)

previsoes_knn = modelo_knn.predict(x_test)

print("KNN Classifier Treinado hehe")

#%%
# Acurácia do modelo KNN

accuracy_knn = accuracy_score(y_test, previsoes_knn)

print(f"Acurácia do modelo KNN Classifier: {accuracy_knn * 100:.2f}%")

#%%
# Comparando os modelos

resultados = pd.DataFrame({
    'Modelo': ['Decision Tree', 'Random Forest', 'KNN'],
    'Acurácia': [accuracy_tree * 100, accuracy_rf * 100, accuracy_knn * 100]
})

print(resultados)

#%% 
# Gráfico de Comparação

plt.figure(figsize=(8,5))
sns.barplot(x='Modelo', y='Acurácia', data=resultados)

plt.ylim(0,1)
plt.title("Comparação de Acurácia entre Modelos")
plt.ylabel("Acurácia (%)")

plt.show()

#%%
# Classification Report para o melhor modelo

print(classification_report(y_test, previsoes_rf, target_names=wine.target_names))

#%%

# Validação Cruzada

scores = cross_val_score(modelo_rf, x, y, cv=5)

print(f"Resultados das 5 folds de validação cruzada: {scores}")

print(f"Média da acurácia: {scores.mean():.2f}")

#%%

# Previsão da Amostra

amostra = x_test.iloc[[0]]

previsao = modelo_rf.predict(amostra)[0]

valor_real = y_test.iloc[0]

print("Classe real:", wine.target_names[valor_real])
print("Classe prevista:", wine.target_names[previsao])