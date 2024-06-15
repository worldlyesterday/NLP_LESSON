from dataGenerator import create_dataloader
from seq2Seq import Seq2Seq, Encoder, Decoder
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from dataGenerator import tokenize
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[:, :-1].to(device)
        trg = batch[:, 1:].to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[:, :-1].to(device)
            trg = batch[:, 1:].to(device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def predict(seq2seq_model, dictionary, src, max_len=50):
    seq2seq_model.eval()
    
    # 将输入句子分词并转换为索引
    tokens = tokenize([src])[0]
    # 假设 'dictionary' 已经被正确初始化，并且包含 '<unk>'
    input_indices = [dictionary.token2id.get(token, dictionary.token2id.get('<unk>', -1)) for token in tokens]
    src = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(1)
    
    # 获取编码器的隐藏状态和细胞状态
    with torch.no_grad():
        hidden, cell = seq2seq_model.encoder(src)
    
    # 初始化解码器的输入
    input_token = dictionary.token2id.get('<sos>', -1)
    input = torch.tensor([input_token], dtype=torch.long, device=device)
    
    outputs = []
    
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = seq2seq_model.decoder(input, hidden, cell)
        
        top1 = output.argmax(1)
        outputs.append(top1.item())
        
        # 如果预测到结束标志，则停止生成
        if top1.item() == dictionary.token2id.get('<eos>', -1):
            break
        
        # 下一步输入是当前时间步的输出
        input = top1
    
    # 将索引转换回词汇
    output_tokens = [dictionary[idx] for idx in outputs]
    
    return output_tokens

def train_seq2seq_model(folder_path):
    dataset, dictionary = create_dataloader(folder_path)
    # 设定训练集和测试集的比例: 80% 用于训练，20% 用于测试
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 使用 random_split 分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_dim = len(dictionary)
    output_dim = len(dictionary)
    enc_emb_dim = dec_emb_dim = 128
    hid_dim = 64
    n_layers = 2
    enc_dropout = dec_dropout = 0.1
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)

    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token2id.get('<pad>'))

    N_EPOCHS = 100
    CLIP = 1

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, test_loader, criterion)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')

    # 测试预测
    sample_sentence = "一阵轻柔婉转的歌声，飘在烟水蒙蒙的湖面上。歌声发自一艘小船之中，船里五个少女和歌嘻笑，荡舟采莲。"
    print("Predicted text:", predict(model, dictionary, sample_sentence))

train_seq2seq_model('./dataset_copy')
