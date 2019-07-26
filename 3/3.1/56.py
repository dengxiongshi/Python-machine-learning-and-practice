# python3
# Description:
# Author:       xiaoshi
# Time:         2019/7/24 9:53
"""使用CountVectorizer并且不去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试"""
# 从sklearn.datasets里导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
# 采用默认的配置对CountVectorizer进行初始化(默认配置不去除英文停用词)，并且赋值给变量count_vec
count_vec = CountVectorizer()

# 只使用词频统计的方式将原始训练和测试文本转化为特征向量。
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

# 从sklearn.naive_bayes里导入朴素贝叶斯分类器。
from sklearn.naive_bayes import MultinomialNB
# 使用默认配置对分类器进行初始化。
mnb_count = MultinomialNB()
# 使用朴素贝叶斯分类器，对CountVectorizer(不去除停用词)后的训练样本进行参数学习。
mnb_count.fit(X_count_train, y_train)

# 输出模型准确性结果。
print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer without filtering stopwords):', mnb_count.score(X_count_test, y_test))
#将分类预测的结果储存在变量y_count_predict中。
y_count_predict = mnb_count.predict(X_count_test)
# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
# 输出更加详细的其他评价分类性能的指标。
print(classification_report(y_test, y_count_predict, target_names=news.target_names))


"""使用TfidfVectorizer并且不去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试"""
from sklearn.feature_extraction.text import TfidfVectorizer
# 采用默认的配置对TfidfVectorizer进行初始化(默认配置不去除英文停用词)，并且赋值给变量tfidf_vec
tfidf_vec = TfidfVectorizer()

# 使用tfidf的方式，将原始训练和测试文本转化为特征向量。
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

# 依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试数据上，对新的特征量化方式进行性能评估。
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):', mnb_tfidf.score(X_tfidf_test, y_test))
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))


"""分别使用CountVectorizer与TfidfVectorizer，并且去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试"""
# 分别使用停用词过滤配置初始化CountVectorizer与TfidfVectorizer。
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

# 使用带有停用词过滤的CountVectorizer对训练和测试文本进行量化处理。
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
# 使用带有停用词过滤的TfidfVectorizer对训练和测试文本进行量化处理。
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

# 初始化默认配置的朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确性评估。
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer without filtering stopwords):', mnb_count_filter.score(X_count_filter_test, y_test))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)

# 初始化默认配置的朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确性评估。
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):', mnb_tfidf_filter.score(X_tfidf_filter_test, y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)

# 对上述两个模型进行更加详细的性能评估。
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))
