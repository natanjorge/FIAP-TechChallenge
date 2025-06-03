import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import numpy as np

# Leitura e pré-processamento
df = pd.read_csv('insurance.csv')
df = df.dropna().drop_duplicates()
df = df[(df['age'] > 0) & (df['bmi'] > 0) & (df['charges'] > 0)]
df = df[~((df['age'] < 18) & (df['smoker'] == 'yes'))]

# Mantém a versão original para gráficos
df_vis = df.copy()  

# Codificação
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Features
X = df.drop('charges', axis=1)
y = df['charges']

# Separação e escala
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputação de valores ausentes
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos
modelos = {
    'Regressão Linear': LinearRegression(),
    'Árvore de Decisão': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

resultados = []

# Validação cruzada para modelos clássicos
for nome, modelo in modelos.items():
    mse = -cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    resultados.append({'Modelo': nome, 'MSE': mse, 'R²': r2})

# GridSearchCV para KNN
param_grid = {'n_neighbors': list(range(1, 21))}
knn = KNeighborsRegressor()
grid_knn = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_knn.fit(X_train_scaled, y_train)

melhor_k = grid_knn.best_params_['n_neighbors']
melhor_knn = grid_knn.best_estimator_
melhor_mse = -grid_knn.best_score_
melhor_r2 = cross_val_score(melhor_knn, X_train_scaled, y_train, cv=5, scoring='r2').mean()
resultados.append({'Modelo': f'KNN (k={melhor_k})', 'MSE': melhor_mse, 'R²': melhor_r2})

# Exibir resultados tabulados
df_resultados = pd.DataFrame(resultados).sort_values(by='R²', ascending=False)
print("\nResultados dos modelos com Validação Cruzada (5-fold):")
print(df_resultados)

# Visualização Comparativa
plt.figure(figsize=(10, 6))
sns.barplot(data=df_resultados, x='Modelo', y='R²', hue='Modelo', palette='viridis', legend=False)
plt.title('Comparação de R² entre os modelos (Validação Cruzada)')
plt.ylabel('R² Médio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_resultados, x='Modelo', y='MSE', hue='Modelo', palette='rocket', legend=False)
plt.title('Comparação de MSE entre os modelos (Validação Cruzada)')
plt.ylabel('MSE Médio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.pairplot(df_vis, vars=['bmi', 'charges'], hue='smoker', palette='Set2', diag_kind='hist')
plt.suptitle('Distribuição e Relação: Fumantes vs BMI vs Charges', y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df_vis, x='smoker', y='charges', hue='sex', palette='pastel')
plt.title('Distribuição dos Custos Médicos por Sexo e Tabagismo')
plt.xlabel('Fumante')
plt.ylabel('Charges')
plt.show()

g = sns.FacetGrid(df_vis, col='region', hue='sex', height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='age', y='charges', alpha=0.7)  
g.add_legend(title='Sexo')
g.set_axis_labels("Idade", "Charges")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Charges vs. Idade por Sexo e Região")
plt.show()

# Comparativo charges por filhos
df_vis['tem_filhos'] = df_vis['children'].apply(lambda x: 'Sem filhos' if x == 0 else 'Com filhos')

plt.figure(figsize=(8, 6))
sns.barplot(data=df_vis, x='tem_filhos', y='charges', hue='tem_filhos', estimator=np.mean, palette='Set2', legend=False)
plt.title('Média de Custos Médicos: Com e Sem Filhos')
plt.ylabel('Média de Charges')
plt.xlabel('Situação Parental')
plt.tight_layout()
plt.show()

# Gráfico de distribuição:
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_vis, x='tem_filhos', y='charges', hue='tem_filhos', palette='coolwarm', legend=False)
plt.title('Distribuição de Custos Médicos: Com e Sem Filhos')
plt.ylabel('Charges')
plt.xlabel('Situação Parental')
plt.tight_layout()
plt.show()
