# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/29 17:21
"""使用skflow内置的LinearRegressor、DNN以及Scikit-learn中的集成回归模型对”美国波士顿房价“数据进行回归预测"""
from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import train_test_split

# 从读取房价数据存储在变量boston中
boston = datasets.load_boston()
# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test =train_test_split(boston.data, boston.target, test_size =0.25, random_state=33)

# 对数据特征进行标准化处理。
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from skflow import TensorFlowLinearRegressor
# 使用skflow的LinearRegressor。
tf_lr = TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_predict = tf_lr.predict(X_test)

# 输出skflow中LinearRegressor模型的回归性能。
print('The mean absolute error of Tensorflow Linear Regressor on boston datasets is', metrics.mean_absolute_error(tf_lr_y_predict, y_test))
print('The mean squared error of Tensorflow Linear Regressor on boston datasets is', metrics.mean_squared_error(tf_lr_y_predict, y_test))
print('The R-squared value of Tensorflow Linear Regressor on boston datasets is', metrics.r2_score(tf_lr_y_predict, y_test))