import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import ast

# 加载数据
data_path = 'word.csv'
data = pd.read_csv(data_path, encoding='utf-8-sig')

# 数据预处理：将data列中的字符串列表转换为实际的列表
data['data'] = data['data'].apply(ast.literal_eval)

# 提取数据和标签
X = data['data'].tolist()
y = np.array(data['label'].tolist())

# 构建字典和语料库
dictionary = Dictionary(X)
corpus = [dictionary.doc2bow(text) for text in X]

# 训练LDA模型
T = 100  # 主题数量可以根据需要调整
lda = LdaModel(corpus, num_topics=T, id2word=dictionary, passes=10)

# 提取特征向量（每个段落的主题分布）
features = np.array([[tup[1] for tup in lda.get_document_topics(bow, minimum_probability=0.0)] for bow in corpus])

# 准备分类器和交叉验证
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 进行交叉验证
results = []
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    results.append(accuracy)

print("平均准确率：", np.mean(results))
