# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/22 18:19
"""利用轮廓系数评价不同类簇数量的K-means聚类实例"""
import numpy as np
from sklearn.cluster import KMeans
# 从sklearn.metrics导入silhouette_score用于计算轮廓系数。
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 分割出3*2=6个子图，并在1号子图作图。
plt.subplot(3, 2, 1)
# 初始化原始数据点。
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X= np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# 在1号子图做出原始数据点阵的分布。
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)

    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')

    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)

    # 绘制轮廓系数与不同类簇数量的直观显示图
    plt.title('K=%s, silhouette coefficient=%0.03f' % (t, sc_score))
    # plt.figure()
    # plt.show()

# 绘制轮廓系数与不同类簇数量的关系曲线
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()