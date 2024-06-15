from dataGenerator import create_dataloader
from transformer import TransformerModel
import torch
import torch.nn as nn
import torch.optim as optim
from dataGenerator import tokenize
from torch.utils.data import random_split, DataLoader
import math

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 训练函数
def train(model, data_loader, criterion, optimizer, ntokens, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        src = batch.to(device)
        src_mask = model._generate_square_subsequent_mask(src.size(0)).to(device)
        optimizer.zero_grad()
        output = model(src, src_mask)
        loss = criterion(output.view(-1, ntokens), src.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 评估函数
def evaluate(model, data_loader, criterion, ntokens, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src = batch.to(device)
            src_mask = model._generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, src_mask)
            loss = criterion(output.view(-1, ntokens), src.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# 预测函数
def predict(transformer_model, dictionary, src, device, max_len=50):
    transformer_model.eval()
    
    # 将输入句子分词并转换为索引
    tokens = tokenize([src])[0]
    # 假设 'dictionary' 已经被正确初始化，并且包含 '<unk>'
    input_indices = [dictionary.token2id.get(token, dictionary.token2id.get('<unk>', -1)) for token in tokens]
    src = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, batch_size)
    
    # 创建源序列掩码
    src_mask = transformer_model._generate_square_subsequent_mask(src.size(0)).to(device)
    
    # 获取编码器的输出
    with torch.no_grad():
        src = transformer_model.encoder(src) * math.sqrt(transformer_model.ninp)
        src = transformer_model.pos_encoder(src)
        memory = transformer_model.transformer.encoder(src, src_mask)
    
    # 初始化解码器的输入
    input_token = dictionary.token2id.get('<sos>', -1)
    input = torch.tensor([[input_token]], dtype=torch.long, device=device)  # (1, 1)
    
    outputs = []
    
    for _ in range(max_len):
        tgt_mask = transformer_model._generate_square_subsequent_mask(input.size(0)).to(device)
        
        with torch.no_grad():
            input = transformer_model.decoder(input) * math.sqrt(transformer_model.ninp)
            input = transformer_model.pos_encoder(input)
            output = transformer_model.transformer.decoder(input, memory, tgt_mask)
            output = transformer_model.decoder(output)
        
        top1 = output[-1, :, :].argmax(1)
        outputs.append(top1.item())
        
        # 如果预测到结束标志，则停止生成
        if top1.item() == dictionary.token2id.get('<eos>', -1):
            break
        
        # 下一步输入是当前时间步的输出
        input = torch.cat([input, top1.unsqueeze(0)], dim=0)
    
    # 将索引转换回词汇
    output_tokens = [dictionary.idx2word[idx] for idx in outputs if idx in dictionary.idx2word]
    
    return output_tokens

# 训练和评估循环
num_epochs = 1
def train_transformer_model(folder_path):
    dataset, dictionary = create_dataloader(folder_path)
    # 设定训练集和测试集的比例: 80% 用于训练，20% 用于测试
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # 定义超参数
    ntokens = len(dictionary)  # 词汇表大小
    ninp = 128  # 词嵌入维度
    nhead = 4  # 注意力头数量
    nhid = 128  # 隐藏层维度
    nlayers = 2  # Transformer 层数
    dropout = 0.5  # Dropout 概率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token2id.get('<pad>'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 使用 random_split 分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer,ntokens, device)
        test_loss = evaluate(model, test_loader, criterion, ntokens,device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # torch.save(model.state_dict(), 'transformer_model.pt')
    # 测试预测
    sample_sentence = "一阵轻柔婉转的歌声，飘在烟水蒙蒙的湖面上。歌声发自一艘小船之中，船里五个少女和歌嘻笑，荡舟采莲。"
    print("Predicted text:", predict(model, dictionary, sample_sentence, device))

train_transformer_model('./dataset_copy')