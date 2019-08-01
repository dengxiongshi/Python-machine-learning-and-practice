# python3
# Description:
# Author:       xiaoshi
# Time:         2019/7/30 9:52
import pandas as pd

train = pd.read_csv('../Datasets/titanic/train.csv')
test = pd.read_csv('../Datasets/titanic/test.csv')
# 先分别输出训练与测试数据的基本信息。
print(train.info())
print(test.info())
# 人工选取对预测有效的特征。
selected_features = ['Pclass', 'Sex', 'SibSp', 'Age', 'Parch', 'Embarked', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

# Embarked特征存在缺失值，需要补完。
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())
# 使用出现频率最高的特征值来填充
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
# 对于Age这种数值类型的特征，使用求平均值或者中位数来填充缺失值
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
# 重新对处理后的训练和测试数据进行查验，发现一切就绪。
X_train.info()

# 采用DictVectorizer对特征向量化。
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 从sklearn.ensemble中导入RandomForestClassifier。
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
# 从流行工具包xgboost导入XGBClassifier用于处理分类预测问题。
from xgboost import XGBClassifier
xgbc = XGBClassifier()

from sklearn.model_selection import cross_val_score
# 使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier以及XGBClassifier进行性能评估，并获得准确性的得分。
print(cross_val_score(rfc, X_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, X_train, y_train, cv=5).mean())

# 使用默认配置的RandomForestClassifier进行预测操作。
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
# 将默认配置的RandomForestClassifier对测试数据的预测结果存储在文件rfc_submission.csv中。
rfc_submission.to_csv('../Datasets/titanic/rfc_submission.csv', index=False)

# 使用默认配置的XGBClassifier进行预测操作。
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('../Datasets/titanic/xgbc_submission.csv', index=False)

# 使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提高XGBClassifier的预测性能。
from sklearn.model_selection import GridSearchCV
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)
# 查验优化之后的XGBClassifier的超参数配置以及交叉验证的准确性。
print(gs.best_score_)
print(gs.best_params_)

# 使用经过优化超参数配置的XGBClassifier对测试数据的预测结果存储在文件xgbc_best_submission.csv中。
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('../Datasets/titanic/xgbc_best_submission.csv', index=False)