import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Leer la base de datos
df = pd.read_csv('COVID19MEXICO.csv')

# Seleccionar las columnas relevantes para el clustering
columnas_relevantes = ['EDAD', 'HIPERTENSION', 'OBESIDAD']
df = df[columnas_relevantes].dropna()

# Convertir a formato numpy array
X = df.to_numpy()

# Lista para guardar la suma de los errores cuadráticos (inercia)
inertia = []

# Aplicar K-Means para diferentes valores de k (número de clusters)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graficar la inercia vs. el número de clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Regla del Codo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.savefig('regla_del_codo.png')
plt.show()

# Encontrar el codo en la gráfica de inercia para determinar el número óptimo de clusters
optimal_k = 4  # Puedes ajustar este valor basándote en la gráfica de la Regla del Codo

# Aplicar K-Means con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Crear un DataFrame con las etiquetas de los clusters
df_clusters = df.copy()
df_clusters['Cluster'] = labels

# Guardar el DataFrame con los clusters en un CSV
df_clusters.to_csv('clustered_COVID19MEXICO.csv', index=False)

# Guardar los centroides en un archivo CSV
centroids_df = pd.DataFrame(centroids, columns=columnas_relevantes)
centroids_df.to_csv('centroids.csv', index=False)

# Visualizar los clusters en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colores para los clusters
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']

for i in range(optimal_k):
    points = X[labels == i]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c=colors[i], label=f'Cluster {i+1}')

# Visualizar los centroides
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='black', marker='X', label='Centroids')

ax.set_title('Clusters 3D')
ax.set_xlabel('EDAD')
ax.set_ylabel('HIPERTENSION')
ax.set_zlabel('OBESIDAD')
ax.legend()
plt.savefig('clusters_3D.png')
plt.show()
