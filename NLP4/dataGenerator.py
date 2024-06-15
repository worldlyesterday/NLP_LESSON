import os
import re
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary

def read_files_from_folder(folder_path):
    total_text = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='ANSI') as file:
                corpus = file.read()
                r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
                corpus = re.sub(r1, '', corpus)
                corpus = re.sub(r'\n|\u3000|本书来自免费小说下载站|更多更新免费电子书请关注', '', corpus)
                corpus = re.sub(r'[^\u4e00-\u9fff]', '', corpus)
                corpus = corpus.replace(" ", "")
                total_text.append(corpus)
    return total_text

def tokenize(text_list):
    with open('./stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    
    tokenized_texts = []
    for text in text_list:
        # 对每个文本进行分词并过滤停用词
        tokens = [token for token in jieba.lcut(text) if token not in stop_words]
        tokenized_texts.append(tokens)
    
    return tokenized_texts

def build_vocab(tokens, min_freq=2):
    # 创建词典实例
    dictionary = Dictionary(tokens)
    
    # 添加特殊标记
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    dictionary.add_documents([[token] for token in special_tokens])  # 添加特殊标记

    # 过滤掉出现频次极低的词汇
    dictionary.filter_extremes(no_below=min_freq, keep_n=None)
    dictionary.compactify()  # 重新分配 id，使其连续

    return dictionary

def text_to_indices(tokens_list, dictionary):
    # 初始化一个空列表用于存储所有文本的索引列表
    indices_list = []

    # 遍历每个文本的词汇列表
    for tokens in tokens_list:
        # 将每个词转换为索引
        indices = [dictionary.token2id[token] if token in dictionary.token2id else dictionary.token2id.get('<unk>') for token in tokens]
        # 将索引列表添加到总的列表中
        indices_list.append(indices)
    
    return indices_list

class TextDataset(Dataset):
    def __init__(self, indices_list, vocab, max_len=50):
        self.vocab = vocab
        self.max_len = max_len
        self.subsequences = []

        # 遍历每个文本的索引列表
        for indices in indices_list:
            # 如果当前文本长度小于max_len，则填充
            if len(indices) < max_len:
                padded_indices = indices + [self.vocab['<pad>']] * (max_len - len(indices))
                self.subsequences.append(padded_indices)
            else:
                # 如果当前文本长度大于或等于max_len，进行切片处理
                self.subsequences.extend([indices[i:i+max_len] for i in range(0, len(indices), max_len) if len(indices[i:i+max_len]) == max_len])

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        subseq = self.subsequences[idx]
        if len(subseq) != self.max_len:
            raise ValueError('Invalid subsequence length: {}'.format(len(subseq)))
        return torch.tensor(subseq, dtype=torch.long)

def create_dataloader(folder_path, batch_size=32, max_len=50):
    texts = read_files_from_folder(folder_path)
    tokenized_texts = tokenize(texts)
    dictionary = build_vocab(tokenized_texts)
    indices_texts = text_to_indices(tokenized_texts, dictionary)
    dataset = TextDataset(indices_texts, dictionary, max_len)
    dictionary.save("./dictionary.bin")
    return dataset, dictionary