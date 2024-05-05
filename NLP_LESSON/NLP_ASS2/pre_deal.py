import jieba
import jieba.analyse
import random
import csv
import re


def read_file(filename, is_word=False):
    # 默认为类名
    target = "data/" + filename + ".txt"
    # 读取文件
    with open(target, "r", encoding='gbk', errors='ignore') as f:
        data = f.read()
        data = data.replace(
            '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = re.sub(r'[^\u4e00-\u9fa5]', '', data)
        f.close()
    # 读取停用词表，并分词
    with open("cn_stopwords.txt", "r", encoding='utf-8') as fp:
        stop_word = fp.read().split('\n')
        fp.close()
    # 分词
    split_word = []
    # 以字为单位
    if is_word:
        for word in data:
            if (word not in stop_word) and (not word.isspace()):
                split_word.append(word)
    # 以词为单位
    else:
        for words in jieba.cut(data):
            if (words not in stop_word) and (not words.isspace()):
                split_word.append(words)
    return split_word


def extract_paragraph(is_word, K=500):
    """
    
    根据每篇文章的长度权重随机抽取段落，总计抽取1000个段落。

    Args:
    is_word (bool): True表示按词提取，False表示按字符提取。
    K (int): 每个段落包含的词数或字符数。
    
    """
    with open("data/inf.txt", "r", encoding='gbk') as f:
        txt_list = f.read().strip().split(',')
    
    total_len = 0
    texts = {}
    segments = {}  # 存储每篇文章的段落分割
    weights = []   # 每篇文章的权重列表

    # 读取文章并计算总长度
    for name in txt_list:
        data = read_file(name, is_word)
        texts[name] = data
        total_len += len(data)

    # 计算每篇文章的权重并预分段
    for name in texts:
        weight = len(texts[name]) / total_len
        weights.append(weight)
        # 预分段
        segments[name] = [texts[name][i:i + K] for i in range(0, len(texts[name]), K)]

    selected_segments = []
    attempts = 0

    while len(selected_segments) < 1000 and attempts < 10000:
        # 根据权重随机选择一篇文章
        selected_name = random.choices(txt_list, weights=weights, k=1)[0]
        available_segments = segments[selected_name]

        if available_segments:
            # 从选中的文章中随机选择一个段落
            segment = random.choice(available_segments)
            selected_segments.append({
                'number': len(selected_segments) + 1,
                'label': selected_name,
                'data': segment
            })
            # 确保不重复选择同一个段落
            available_segments.remove(segment)
        attempts += 1

    save_path = 'char.csv' if is_word else 'word.csv'
    with open(save_path, 'w', newline='', encoding='utf-8') as fp:
        fieldnames = ['number', 'label', 'data']
        csv_writer = csv.DictWriter(fp, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(selected_segments)


if __name__ == "__main__":
    # 以字为单位抽取段落
    extract_paragraph(True, K=1000)
    # 以词为单位抽取段落
    extract_paragraph(False, K=1000)
