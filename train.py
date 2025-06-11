import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TextClassifier
from dataloader import get_data_loader
import torch.nn as nn
import torch.nn.functional as F


# 超参数设置
VOCAB_SIZE = 10000  # 假设词汇表大小为10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 2  # 两个类别：人类和大模型生成
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# 加载数据
train_loader, test_loader = get_data_loader(BATCH_SIZE)

# 初始化模型、损失函数和优化器
model = TextClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
model.train()
for epoch in range(NUM_EPOCHS):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')