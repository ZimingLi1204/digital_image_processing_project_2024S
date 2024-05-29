import torch
from sklearn.cluster import KMeans
import numpy as np

# 示例数据
data = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

# 创建K-means对象，并设置初始化为K-means++
kmeans = KMeans(n_clusters=3, init="k-means++")

# 执行聚类
kmeans.fit(data)

# 输出聚类结果
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Final centroids:")
print(centroids)
print("Assigned labels:")
print(labels)


def auto_kmeans(data):
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=3, init="k-means++")
        kmeans.fit(data)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

    pass