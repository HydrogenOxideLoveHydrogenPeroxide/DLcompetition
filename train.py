import torch
from torch import nn
import numpy
import random
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
from dataloader import get_data_loader
from model import Classifier
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datetime
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备：{device}')

def set_seed(seed=42):
    random.seed(seed)                      # Python 随机数
    np.random.seed(seed)                   # NumPy 随机数
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # 多卡 GPU
    # torch.backends.cudnn.deterministic = True   # 固定算法
    # torch.backends.cudnn.benchmark = False      # 禁用性能自动优化

# 参数设置
MODEL_NAME ="Qwen/Qwen2.5-0.5B" #"bert-base-uncased"#MODEL_ROOT_DIR/"bert-base-uncased"# #备选"prajjwal1/bert-tiny"
MODEL_PATH_DIR=Path(os.getcwd())/'result'
os.path.exists(MODEL_PATH_DIR) or os.makedirs(MODEL_PATH_DIR)
MODEL_PATH = "result/best_model.pth"       # 训练阶段保存的模型路径
EPOCHS = 20
BATCH_SIZE = 8  # Qwen2.5 体积大，减小 batch size 防止 OOM
LR = 2e-5
MAX_LEN = 512
SEED=42
set_seed(SEED)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, test_loader = get_data_loader(BATCH_SIZE)

# 编码函数
def encode_batch(batch):
    texts, labels = batch
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    return encodings['input_ids'], encodings['attention_mask'], labels.detach().clone()

# 初始化模型
model = Classifier(model_name=MODEL_NAME).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# 训练函数
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_mask, labels = encode_batch(batch)
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

# 评估函数
def evaluate():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = encode_batch(batch)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    return precision,recall,f1
  
  
def log_result(result):
    with open(f'{datetime}_train.json') as logf:
        json.dump(result,logf)
        
if __name__ == "__main__":
    train()
    precision,recall,f1=evaluate()
    train_log={
        'args':{
            'model_name':MODEL_NAME,
            'epoch':EPOCHS,
            'batch_size':BATCH_SIZE,
            'lr':LR,
            'max_len':MAX_LEN,
            'seed':SEED
        },
        'result':[precision,recall,f1]
    }
    log_result(train_log)