
# Previsão de Custos Médicos com Regressão

Este projeto faz parte do **Tech Challenge da Fase 1** do IADT e tem como objetivo desenvolver um modelo preditivo de regressão para estimar os custos médicos individuais com base em variáveis demográficas e comportamentais.

## Introdução

Prever os custos médicos individuais é uma tarefa importante para operadoras de saúde e seguradoras, permitindo uma precificação mais justa e baseada em risco. Este projeto explora técnicas de regressão para estimar esses custos com base em variáveis demográficas e comportamentais.

## Objetivo

Criar, treinar e avaliar modelos de regressão capazes de prever com precisão o valor dos encargos cobrados pelo seguro de saúde com base em características como:

- Idade
- Sexo
- IMC (Índice de Massa Corporal)
- Número de filhos
- Tabagismo
- Região

## Dataset

O conjunto de dados utilizado é baseado no clássico `insurance.csv`, contendo as respectivas colunas:

```
age,sex,bmi,children,smoker,region,charges
```

Pré-processamos os dados para remover valores inconsistentes, duplicados e registros com características não realistas (por exemplo, fumantes menores de idade).

## Pré-processamento

- Remoção de `NaN` e duplicatas.
- Exclusão de outliers e dados inválidos.
- Codificação de variáveis categóricas (`sex`, `smoker`, `region`).
- Imputação de valores ausentes.
- Normalização dos dados com `StandardScaler`.

## Modelos Treinados

Utilizamos os seguintes algoritmos de regressão:

- Regressão Linear
- Árvore de Decisão
- Random Forest
- Gradient Boosting
- KNN (com `GridSearchCV` para encontrar o melhor valor de `k`)

Todos os modelos foram avaliados com **validação cruzada (5-fold)**.

## Avaliação

As métricas utilizadas para avaliação dos modelos foram:

- R² (coeficiente de determinação)
- MSE (Erro Quadrático Médio)

### Comparação de Desempenho (validação cruzada):

| Modelo            | R² Médio | MSE Médio    |
| ----------------- | -------- | ------------ |
| Gradient Boosting | 0.834665 | 2.270612e+07 |
| Random Forest     | 0.820113 | 2.465886e+07 |
| KNN (k=7)         | 0.769987 | 3.181062e+07 |
| Regressão Linear  | 0.725757 | 3.753124e+07 |
| Árvore de Decisão | 0.669382 | 4.490145e+07 |


## Visualizações

O projeto inclui diversas visualizações para auxiliar na interpretação dos dados e resultados:

- Pairplots de IMC vs. Encargos (dividido por tabagismo)
- Boxplots de `charges` por sexo e hábito de fumar
- Scatterplots de idade vs. encargos por região
- Comparações de custos entre pessoas com e sem filhos
- Gráficos de barras comparativos de MSE e R² entre modelos

## Entregável

- Código-fonte
- Visualizações e análises
- [Vídeo explicativo no YouTube](https://youtu.be/rBz1FxwuZvA)

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script principal:
   ```bash
   python projeto_regressao.py
   ```

## Bibliotecas Utilizadas

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn