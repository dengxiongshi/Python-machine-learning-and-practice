# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/17 10:17
# 从sklearn.datesets里导入手写体数字加载器
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits()
# 检视数据规模和特征维度
print(digits.data.shape)

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
# 从sklearn.cross_validation中导入train_test_split用于数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# 分别检视训练与测试数据规模
print(y_train.shape)
print(y_test.shape)

# 从sklearn.preprocessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC

# 从仍然需要对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)
# print(y_predict)

# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of Linear SVC is', lsvc.score(X_test, y_test))

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from  sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))