# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/29 9:14
"""对比随机决策树森林以及XGBoost模型对泰坦尼克号上的乘客是否生还的预测能力"""
import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 选取pclass、age以及sex作为训练特征。
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对缺少数据进行填充。
X['age'].fillna(X['age'].mean(), inplace=True)
# 分割数据，依然采用25%用于测试。
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 类别型特征向量化。
from sklearn.feature_extraction import DictVectorizer
dvt = DictVectorizer(sparse=False)
X_train = dvt.fit_transform(X_train.to_dict(orient='record'))
X_test = dvt.transform(X_test.to_dict(orient='record'))

# 采用默认配置的随机森林分类器对测试集进行预测。
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, y_test))

# 采用默认配置的XGBoost模型对相同的测试集进行预测。
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print('The accuray of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))
