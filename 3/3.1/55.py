# python3
# Description:  DictVectorizer对使用字典存储的数据进行特征抽取与向量化
# Author:       xiaoshi
# Time:         2019/7/24 9:07
# 定义一组字典列表，用来表示多个数据样本(每个字典代表一个数据样本)。
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
# 从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# 初始化DictVectorizer特征抽取器
dvt = DictVectorizer()
# 输出转化之后的特征矩阵
print(dvt.fit_transform(measurements).toarray())
# 输出各个维度的特征含义
print(dvt.get_feature_names())