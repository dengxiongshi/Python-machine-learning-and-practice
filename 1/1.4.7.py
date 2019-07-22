# python3
# Name:         1.4.7
# Description:  
# Author:       xiaoshi
# Time:         2019/7/11 16:22
# In[1]:


import pandas as pd

df_train = pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-train.csv')  # 读取训练数据集
df_test = pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-test.csv')
# 选取'Clump Thickness'与'Cell Size'作为特征,构建测试集中的正负分类样本
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# In[2]:


# 导入matplotlib工具包中的pyplot并简化命名为plt。
import matplotlib.pyplot as plt

# 绘制图2中的良性肿瘤样本点，标记为红色的o。
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
# 绘制图2中的恶性肿瘤样本点，标记为黑色的x。
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

# 绘制x,y轴的说明。
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# plt.legend('negative','positive')

# 显示图2。
plt.show()

# 导入numpy包
import numpy as np

# 利用numpy中的random函数随机采样直线的截距和系数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制一条随机直线
plt.plot(lx, ly, c='yellow')

# 绘制图
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

# 导入sklearn中的逻辑贝叶斯分类器
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# 使用前10条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
# 函数映射到二维平面
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制图
plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

lr = LogisticRegression()
# 使用所有训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print('Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制图
plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
