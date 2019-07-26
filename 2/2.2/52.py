# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/23 11:41
import pandas as pd
import numpy as np

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 分割训练数据的特征向量和标记
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 从sklearn.decomposition导入PCA。
from sklearn.decomposition import PCA

# 初始化一个可以将高纬度特征向量(六十四维)压缩至二个维度的PCA。
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布。
import matplotlib.pyplot as plt

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

if __name__ == '__main__':
    plot_pca_scatter()