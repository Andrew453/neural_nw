import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных из CSV файла
data = pd.read_csv('./result_dogs', header=None)

# Разделение на параметры и результат
X = data.iloc[:, :4].values  # Первые 4 столбца как параметры
y = data.iloc[:, 4].values    # Пятый столбец как результат

# Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Создание цветовой карты для классов
colors = {0: 'blue', 1: 'green', 2: 'red'}
scatter_colors = [colors[label] for label in y]

# Построение графика
plt.figure(figsize=(10, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scatter_colors, alpha=0.5)
plt.title('Iris-setosa - blue, Iris-versicolor - green,Iris-virginica - red')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()
