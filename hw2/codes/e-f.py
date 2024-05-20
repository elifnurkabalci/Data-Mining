# Libraries
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

df = pd.read_csv('sample_data/worms_64d.txt', sep=' ', header=None)

df.head()

# Fill NaN values with the median of each column
df.fillna(df.median(), inplace=True)

data_array = df.to_numpy()

def calculate_connectivity_and_density(graph, data):
    """
    Her iki alt küme arasındaki bağlantı gücünü ve her alt kümenin yerel yoğunluğunu hesaplar.

    :param graph: Alt kümeler arasındaki ilişkileri temsil eden bir NetworkX grafiği.
    :param data: Numpy array tipinde veri seti.
    :return: Bağlantı güçleri ve yerel yoğunluklar.
    """
    connectivity = {}
    density = {}

    for node in graph.nodes():
        # Yerel Yoğunluk Hesaplama
        neighbors = list(graph.neighbors(node))
        density[node] = len(neighbors)

        # Bağlantı Gücü Hesaplama
        for neighbor in neighbors:
            if (node, neighbor) not in connectivity and (neighbor, node) not in connectivity:
                connectivity[(node, neighbor)] = 1 / euclidean_distances([data[node]], [data[neighbor]])[0][0]

    return connectivity, density


def chameleon_clustering_eucl(data, initial_clusters=5, final_clusters=3):
    """
    Chameleon clustering algoritması uygulaması.

    :param data: Numpy array tipinde veri seti.
    :param initial_clusters: İlk aşamada kullanılacak küme sayısı.
    :param final_clusters: Son aşamada oluşturulacak küme sayısı.
    :return: Son kümelenmiş veri.
    """
    # Adım 1: Veri Bölümleme
    initial_kmeans = KMeans(n_clusters=initial_clusters).fit(data)
    initial_labels = initial_kmeans.labels_

    # Adım 2: Graf Oluşturma
    graph = nx.Graph()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if initial_labels[i] == initial_labels[j]:
                distance = euclidean_distances([data[i]], [data[j]])[0][0]
                graph.add_edge(i, j, weight=distance)

    # Adım 3: Bağlantı Gücü ve Yerel Yoğunluk Hesaplama
    connectivity, density = calculate_connectivity_and_density(graph, data)
    # connectivity ve density, hesaplanan bağlantı güçleri ve yerel yoğunlukları içerir.

    # Adım 4: Benzer Alt Kümeleri Birleştirme
    # Hiyerarşik kümeleme için Euclidean mesafelerini kullanarak bir mesafe matrisi oluştur
    distance_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if j in graph[i]:  # Kenarın varlığını kontrol et
                distance_matrix[i, j] = distance_matrix[j, i] = graph[i][j]['weight']


    linkages = linkage(distance_matrix, method='average')
    hierarchy_clusters = fcluster(linkages, final_clusters, criterion='maxclust')

    # Adım 5: Son Kümeleri Oluşturma
    return hierarchy_clusters

def chameleon_clustering_cdist(data, initial_clusters=5, final_clusters=3, distance_metric='cityblock'):
    """
    Chameleon clustering algoritması uygulaması.

    :param data: Numpy array tipinde veri seti.
    :param initial_clusters: İlk aşamada kullanılacak küme sayısı.
    :param final_clusters: Son aşamada oluşturulacak küme sayısı.
    :return: Son kümelenmiş veri.
    """
    # Adım 1: Veri Bölümleme
    initial_kmeans = KMeans(n_clusters=initial_clusters).fit(data)
    initial_labels = initial_kmeans.labels_

    # Adım 2: Graf Oluşturma
    graph = nx.Graph()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if initial_labels[i] == initial_labels[j]:
                distance = cdist([data[i]], [data[j]], metric=distance_metric)[0][0]
                graph.add_edge(i, j, weight=distance)

    # Adım 3: Bağlantı Gücü ve Yerel Yoğunluk Hesaplama
    connectivity, density = calculate_connectivity_and_density(graph, data)
    # connectivity ve density, hesaplanan bağlantı güçleri ve yerel yoğunlukları içerir.

    # Adım 4: Benzer Alt Kümeleri Birleştirme
    distance_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if graph.has_edge(i, j):
                distance_matrix[i, j] = distance_matrix[j, i] = graph[i][j]['weight']


    linkages = linkage(distance_matrix, method='average')
    hierarchy_clusters = fcluster(linkages, final_clusters, criterion='maxclust')

    # Adım 5: Son Kümeleri Oluşturma
    return hierarchy_clusters

clusters = chameleon_clustering_eucl(data_array)
# Burada clusters, son kümelenmiş veriyi içerir. İhtiyaca göre görselleştirme veya başka işlemler yapılabilir.

clusters2 = chameleon_clustering_cdist(data_array)

# Sonuçların görselleştirilmesi
plt.scatter(data_array[:, 0], data_array[:, 1], c=clusters, cmap='viridis')
plt.title('Chameleon Clustering Result with Eucl')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Sonuçların görselleştirilmesi
plt.scatter(data_array[:, 0], data_array[:, 1], c=clusters2, cmap='viridis')
plt.title('Chameleon Clustering Result with Cdist')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()