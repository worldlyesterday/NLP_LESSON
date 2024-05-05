import jieba
from collections import Counter
import math
import os
import re 
import matplotlib.pyplot as plt
# 步骤1: 预处理文本
def preprocessText(text, stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        # 去除每行两端的空白字符并转换为集合
        stopwords = set([line.strip() for line in file])
    # 去除标点符号等非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba分词
    cutWords = jieba.cut(text)
    # 过滤停用词
    filtered_words = [word for word in cutWords if word not in stopwords]
    filtered_word = [word for word in text if word not in stopwords]
    return filtered_word , filtered_words

# 步骤2: 对文件夹中的所有文本进行处理并统计词频
def processTexts(text_path, stopwords):
    unigram_tf_word = Counter()
    unigram_tf_words = Counter()
    # 文件内容用ansi编码形式导入
    with open(text_path, 'r', encoding='ansi') as file:
        text = file.read()
        word , words = preprocessText(text, stopwords)
        # 字
        unigram_tf_word.update(word)
        bigram_tf_word = Counter([(word[i], word[i + 1]) for i in range(len(word) - 1)])    
        trigram_tf_word = Counter([(word[i], word[i + 1], word[i + 2]) for i in range(len(word) - 2)])
        # 词
        unigram_tf_words.update(words)
        bigram_tf_words= Counter([(words[i], words[i + 1]) for i in range(len(words) - 1)])    
        trigram_tf_words = Counter([(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)])
    return unigram_tf_word, bigram_tf_word, trigram_tf_word, unigram_tf_words, bigram_tf_words, trigram_tf_words

def calc_entropy_unigram(name,unigram_tf, is_ci=0):
    # 计算一元模型的信息熵
    word_len = sum([item[1] for item in unigram_tf.items()])
    entropy = sum(
        [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
            unigram_tf.items()])
    if is_ci:
        print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

def calc_entropy_bigram(name, unigram_tf, bigram_tf, is_ci=0):
    bigram_len = sum([item[1] for item in bigram_tf.items()])
    entropy = []
    for bigram in bigram_tf.items():
        p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
        p_x_y = bigram[1] / unigram_tf[bigram[0][0]]  # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
    entropy = sum(entropy)
    if is_ci:
        print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

def calc_entropy_trigram(name,bigram_tf,trigram_tf, is_ci):
    trigram_len = sum([item[1] for item in trigram_tf.items()])
    entropy = []
    for trigram in trigram_tf.items():
        p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
        p_x_y = trigram[1] / bigram_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
    entropy = sum(entropy)
    if is_ci:
        print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

def my_plot(X, Y1, Y2, Y3, num):
    # 设置字体为中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # 标签位置
    x = range(0, len(X))
    # 柱状图宽度
    width = 0.2
    # 各柱状图位置
    x1_width = [i - width * 2 for i in x]
    x2_width = [i - width for i in x]
    x3_width = [i for i in x]
    # 设置图片大小、绘制柱状图
    plt.figure(figsize=(19.2, 10.8))
    plt.bar(x1_width, Y1, fc="r", width=width, label="一元模型")
    plt.bar(x2_width, Y2, fc="b", width=width, label="二元模型")
    plt.bar(x3_width, Y3, fc="g", width=width, label="三元模型")
    # 设置x轴
    plt.xticks(x, X, rotation=40, fontsize=10)
    plt.xlabel('数据', fontsize=10)
    # 设置y轴
    plt.ylabel('信息熵', fontsize=10)
    plt.ylim(0, max(Y1) + 2)
    # 标题
    if (num == 1):
        plt.title("以字为单位的信息熵", fontsize=10)
    elif num == 2:
        plt.title("以词为单位的信息熵", fontsize=10)
    # 标注柱状图上方文字
    autolabel(x1_width, Y1)
    autolabel(x2_width, Y2)
    autolabel(x3_width, Y3)
    text = num == 1 and "字" or "词"
    plt.legend()
    plt.savefig('各语料库信息熵（以' + text + '为单位）' + '.png')
    plt.show()


def autolabel(x, y):
    for a, b in zip(x, y):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=10)
        
if __name__ == "__main__":
    files = []
    uniWord = []
    biWord = []
    triWord = []
    uniWords = []
    biWords = []
    triWords = []
    stopwords_path = 'cn_stopwords.txt'
    folder_path = './textFolder'
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            # 文件名
            file = filename.split('.')[0]
            files.append(file)
            # 计算词频
            unigram_tf_word, bigram_tf_word, trigram_tf_word, unigram_tf_words, bigram_tf_words, trigram_tf_words = processTexts(folder_path + '/' + filename, stopwords_path)
            # 计算一元模型信息熵
            uni = calc_entropy_unigram(file,unigram_tf_word, is_ci=0)
            unis = calc_entropy_unigram(file,unigram_tf_words, is_ci=1)
            uniWord.append(uni)
            uniWords.append(unis)
            # 计算二元模型信息熵
            bi = calc_entropy_bigram(file,unigram_tf_word, bigram_tf_word, is_ci=0)
            bis = calc_entropy_bigram(file,unigram_tf_words, bigram_tf_words, is_ci=1)
            biWord.append(bi)
            biWords.append(bis)
            # 计算三元模型信息熵
            tri = calc_entropy_trigram(file,bigram_tf_word, trigram_tf_word, is_ci=0)
            tris = calc_entropy_trigram(file,bigram_tf_words, trigram_tf_words, is_ci=1)
            triWord.append(tri)
            triWords.append(tris)
    
    my_plot(files, uniWord, biWord, triWord, 1)
    my_plot(files, uniWords, biWords, triWords, 2)

 
