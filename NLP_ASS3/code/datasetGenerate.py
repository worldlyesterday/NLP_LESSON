import re
import jieba

DATA_PATH = '../jyxstxtqj/'


def get_corpus(file_path):
    corpus = ''
    # 去除无用字符
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open('../stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
        f.close()
    # print(stop_words)
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        # 删除无用字符
        corpus = re.sub(r1, '', corpus)
        corpus = re.sub(r'\n|\u3000|本书来自免费小说下载站更多更新免费电子书请关注', '', corpus)
        f.close()
    # jieba分词
    words = list(jieba.cut(corpus))
    return [word for word in words if word not in stop_words]


if __name__ == '__main__':
    with open('../inf.txt', 'r') as inf:
        txt_list = inf.readline().split(',')
        for name in txt_list:
            file_name = name + '.txt'
            file_content = get_corpus(DATA_PATH + file_name)
            temp = []
            count = 0
            lines = []
            for w in file_content:
                if count % 50 == 0:
                    lines.append(" ".join(temp))
                    temp = []
                    count = 0
                temp.append(w)
                count += 1
            with open('../dataset/' + 'trainDataset_' + file_name, 'w', encoding='utf8') as train:
                train.writelines(lines)
