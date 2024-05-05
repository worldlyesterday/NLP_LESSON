import os
import re
import jieba
import matplotlib.pyplot as plt
from collections import Counter

# 加载停用词函数
def loadStopwords(filepath):
    # 停用词用utf-8编码形式导入
    with open(filepath, 'r', encoding='utf-8') as file:
        # 去除每行两端的空白字符并转换为集合
        stopwords = set([line.strip() for line in file])
    return stopwords

# 步骤1: 预处理文本
def preprocessText(text, stopwords):
    # 去除标点符号等非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba分词
    words = jieba.cut(text)
    # 过滤停用词
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words


# 步骤2: 对文件夹中的所有文本进行处理并统计词频
def processTexts(folder_path, stopwords):
    word_counts = Counter()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            # 读取文本文件内容
            file_path = os.path.join(folder_path, filename)
            # 文件内容用ansi编码形式导入
            with open(file_path, 'r', encoding='ansi') as file:
                text = file.read()
                words = preprocessText(text, stopwords)
                word_counts.update(words)
    return word_counts

# 步骤3: 绘制词频-排名图
def plot_zipf_law(word_counts):
    # 设置字体为中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    ranks = range(1, len(word_counts) + 1)
    frequencies = [freq for _, freq in word_counts.most_common()]
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".")
    plt.xlabel('排名')
    plt.ylabel('词频')
    plt.title('Zipf\'s Law')
    plt.savefig('Zipf\'s Law' + '.png')
    plt.show()

if __name__ == "__main__":
    # 停用词文件路径
    stopwords_path = 'cn_stopwords.txt'
    # 语料库文本文件夹路径
    folder_path = './textFolder'
    stopwords = loadStopwords(stopwords_path)
    word_counts = processTexts(folder_path, stopwords)
    plot_zipf_law(word_counts)
