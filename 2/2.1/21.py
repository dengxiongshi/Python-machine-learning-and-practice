# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/18 11:11
# 从sklearn.datasets里导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print(len(news.data))
# print(news.data[0])
print(news.DESCR)

# 从sklearn.model_selection导入train_test_split
from sklearn.model_selection import train_test_split
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 从sklearn.native_bays里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对象模型参数进行估计
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from  sklearn.metrics import classification_report
print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))

