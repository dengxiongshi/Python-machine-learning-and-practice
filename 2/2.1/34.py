# python3
# Description:  线性回归器——美国波士顿地区房价数据描述
# Author:       xiaoshi
# Time:         2019/7/20 16:38
# 从sklearn.datasets导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print(boston.DESCR)
# print(boston.data[0])

# 从sklearn.model_selection导入数据分割
from sklearn.model_selection import train_test_split
import numpy as np

# X = boston.data
# y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size =0.25, random_state=33)

# 分析回归目标值的差异
print('The max target value is', np.max(boston.target))
print('The min target value is', np.min(boston.target))
print('The average target value is', np.mean(boston.target))


# 从sklean.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
# 更改代码错误
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 从sklearn.linear_model导入LinearRegression.
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器LinearRegression
lr = LinearRegression()
# 使用训练数据进行参数估计。
lr.fit(X_train, y_train)
# 对测试数据进行回归预测。
lr_y_predict = lr.predict(X_test)

# 从sklearn.linear_model导入SGDRegression
from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)


"""
使用三种回归评价机制以及两种调用R-squared评价模块的方法，对本节模型的回归性能做出评价。
"""
# 使用LinearRegression模型自带的评估模块，并输出评价结果。
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))

# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 使用r2_score模块，并输出评估结果
print('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))
# 使用mean_squared_error模块，并输出评估结果。
print('The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
# 使用mean_absolute_error模块，并输出评估结果。
print('The mean absolute error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print()

# 使用SGDRegressor模型自带的评估模块，并输出评估结果。
print('The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test))
# 使用r2_score模块，并输出评估结果
print('The value of R-squared of LinearRegression is', r2_score(y_test, sgdr_y_predict))
# 使用mean_squared_error模块，并输出评估结果。
print('The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
# 使用mean_absolute_error模块，并输出评估结果。
print('The mean absolute error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))


"""
使用三种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据做出预测
"""
# 从sklearn.svm中导入支持向量机(回归)模型
from sklearn.svm import SVR

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

"""
对三种核函数配置下的支持向量机回归模型在相同测试集上进行性能评估
"""
# 使用R-squared、MSE和MAE指标对三种配置的支持向量机(回归)模型在相同测试集上进行评估
print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print()

print('R-squared value of linear SVR is', poly_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print()

print('R-squared value of linear SVR is', rbf_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print()


"""
使用两种不同配置的K近邻回归模型对美国波士顿房价数据进行回归预测
"""
# 从sklearn.neighbors导入KNeighbosRegressor(K近邻回归器)
from sklearn.neighbors import KNeighborsRegressor

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

"""
对两种不同配置的K近邻回归模型在美国波士顿房价数据上进行预测性能的评估
"""
# 使用R-squared、MSE以及MAE三种指标对平均回归配置的K近邻模型在测试集上进行性能的评估
print('R-squared value of uniform-weighted KNeighborRegression:', uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighborRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNeighborRegression',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print()

# 使用R-squared、MSE以及MAE三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估
print('R-squared value of uniform-weighted KNeighborRegression:', dis_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighborRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNeighborRegression',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))


"""
使用回归树对美国波士顿房价训练数据进行学习，并对测试数据进行预测
"""
# 从sklearn.tree中导入DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# 使用默认配置初始化DecisionTreeRegressor
dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据建回归树。
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测，并将预测值存储在变量dtr_y_predict
dtr_y_predict = dtr.predict(X_test)

"""对单一回归模型在美国波士顿房价测试数据上的预测性能进行评估"""
# 使用R-squared、MSE以及MAE指标对默认配置的回归树在测试集上进行性能评估
print('R-squared value of DecisionTreeRegression:', dtr.score(X_test, y_test))
print('The mean squared error of DecisionTreeRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absolute error of DecisionTreeRegression',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))


"""使用三种集成回归模型对美国波士顿房价训练数据进行学习，并对测试数据进行预测"""
# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesRegressor以及GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# 使用RandomForestRegressor训练模型，并对测试数据做出预测
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreesRegressor训练模型，并对测试数据做出预测
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

"""对三种集成回归模型在美国波士顿房价测试数据上的回归预测性能进行评估"""
# 使用R-squared、MSE以及MAE指标对默认配置的随机回归树在测试集上进行性能评估
print('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))
print('The mean squared error of RandomForestRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absolute error of RandomForestRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

# 使用R-squared、MSE以及MAE指标对默认配置的极端回归树在测试集上进行性能评估
print('R-squared value of ExtraTreesRegressor:', etr.score(X_test, y_test))
print('The mean squared error of ExtraTreesRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absolute error of ExtraTreesRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度
print(np.sort(list(zip(etr.feature_importances_, boston.feature_names)), axis=0)) # 更改代码，添加list()函数

# 使用R-squared、MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估
print('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostingRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absolute error of GradientBoostingRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
