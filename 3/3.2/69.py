# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/28 9:12
"""使用词袋法(Bag-of-Words)对示例文本进行特征向量化"""
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'
# 从sklearn.feature_extraction.text中导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
sentences = [sent1, sent2]
# 输出特征向量化后的表示。
print(count_vec.fit_transform(sentences).toarray())
# 输出向量各个维度的特征含义。
print(count_vec.get_feature_names())

"""使用NLTK对示例文本进行语言学分析"""
import nltk
# nltk.download('punkt')
# 对句子进行词汇分割和正规化，有些情况如aren't需要分割为are和n't；或者I'm要分割为I和'm。
token_1 = nltk.word_tokenize(sent1)
print(token_1)

token_2 = nltk.word_tokenize(sent2)
print(token_2)
# 整理两句的词表，并且按照ASCII的排序输出。
vocab_1 = sorted(set(token_1))
print(vocab_1)
vocab_2 = sorted(set(token_2))
print(vocab_2)

# 初始化stemmer寻找各个词汇最原始的词根。
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in token_1]
print(stem_1)
stem_2 = [stemmer.stem(t) for t in token_2]
print(stem_2)

# 初始化词性标注器，对每个词汇进行标注。
pos_tag_1 = nltk.tag.pos_tag(token_1)
print(pos_tag_1)
pos_tag_2 = nltk.tag.pos_tag(token_2)
print(pos_tag_2)