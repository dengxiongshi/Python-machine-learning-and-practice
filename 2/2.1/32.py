# python3
# Description:  集成模型对泰坦尼克号乘客是否生还的预测
# Author:       xiaoshi
# Time:         2019/7/20 10:42
import pandas as pd
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

# 根据我们对这场事故的了解，sex、age、pclass这些特征都很有可能是决定幸免与否的关键因素
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这样可以在保证顺利训练模型的同时，尽可能不影响预测任务
X['age'].fillna(X['age'].mean(), inplace=True)

# 数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=True)
# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))


# 使用单一决策树进行模型训练以及预测分析。
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_predict = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_predict = gbc.predict(X_test)

# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report

# 输出单一决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_predict, y_test))

# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print('The accuracy of random forest classifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_predict, y_test))

# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_predict, y_test))