# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/19 10:30
# 从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris

# 使用加载器读取数据并且存入变量iris.
iris = load_iris()

# 查验数据规模
print(iris.data.shape)

# 查看数据说明
print(iris.DESCR)

# 从sklearn.model_selection里选择导入train_test_split用于数据分割
from sklearn.model_selection import train_test_split
# 采取25%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier,即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用K近邻分类器对测试数据进行类别预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))

# 使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=iris.target_names))
