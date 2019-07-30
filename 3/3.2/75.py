# python3
# Description:  使用Tensorflow自定义一个线性分类器用于对“良/恶性乳腺癌肿瘤”进行预测
# Author:       xiaoshi
# Time:         2019/7/29 12:52
import numpy as np
import pandas as pd
import tensorflow as tf

# 从本地使用pandas读取乳腺癌肿瘤的训练和测试数据。
train = pd.read_csv('D:/python-work/Python机器学习及实践/1/Datasets/Breast-Cancer/breast-cancer-train.csv')
test = pd.read_csv('D:/python-work/Python机器学习及实践/1/Datasets/Breast-Cancer/breast-cancer-test.csv')
# 分隔特征与分类目标。
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

# 得以一个tensorflow的变量b作为线性模型的截距，同时设置初始值1.0。
b = tf.Variable(tf.zeros([1]))
# 定义一个tensorflow的变量W作为线性模型的系数，并设置初始值为-1.0至1.0之间均匀分布的随机数。
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

# 显式定义这个线性函数。
y = tf.matmul(W, X_train) + b
# 使用tensorflow中的reduce_mean取得训练集上均方误差。
loss = tf.reduce_mean(tf.square(y - y_train))

# 使用梯度下降估计参数W,b,并且设置迭代步长为0.01，这个与Scikit-learn中的SGDRegressor类似。
optimizer = tf.train.GradientDescentOptimizer(0.01)

# 以最小二乘损失为优化目标。
train = optimizer.minimize(loss)

# 初始化所有变量。
init = tf.global_variables_initializer()
# 开启Tensorflow中的会话。
sess = tf.Session()
# 执行变量初始化操作。
sess.run(init)

# 迭代1000轮次，训练参数。
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# 准备测试样本。
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]
# 以最终更新的参数作图。
import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx = np.arange(0, 12)
# 以0.5作为分界面,最小二乘法
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]
plt.plot(lx, ly, color='green')
plt.show()
