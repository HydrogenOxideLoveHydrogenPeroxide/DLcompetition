import torch
from torch import nn
import numpy as np
import random
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
import datetime
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from model import get_peft_model_for_classification
from dataloader import get_data_loader  # 自定义的数据加载器模块

# ======================== 基础设置 ========================


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n>>> 当前设备：{device}\n")

# 参数配置
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# 结果保存路径
SAVE_ROOT_DIR = Path("results")
SAVE_ROOT_DIR.mkdir(parents=True, exist_ok=True)


# ======================== 随机种子 ========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ======================== Tokenizer ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# ======================== 模型定义 ========================
model = get_peft_model_for_classification(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

train_loader, test_loader = get_data_loader(BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.BCEWithLogitsLoss()

def encode_batch(batch):
    texts, labels = batch
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    return encodings['input_ids'], encodings['attention_mask'], labels.detach().clone()

# ======================== 训练函数 ========================

def train(MODEL_PATH):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_mask, labels = encode_batch(batch)
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n模型已保存至: {MODEL_PATH}")

# ======================== 评估函数 ========================

def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = encode_batch(batch)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)

            all_preds.extend(preds.float().cpu().numpy())  # 这里转换float
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-score:  {f1:.4f}")
    return precision, recall, f1

# ======================== 结果保存 ========================

def log_result(result,log_path):
    with open(log_path, 'w') as f:
        json.dump(result, f, indent=2)

# ======================== 主程序入口 ========================
def train_main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR=SAVE_ROOT_DIR/timestamp
    SAVE_DIR.exists() or SAVE_DIR.mkdir()
    MODEL_PATH = SAVE_DIR / "best_model.pth"
    log_path=SAVE_DIR/'train_result.json'
    
    train(MODEL_PATH)
    precision, recall, f1 = evaluate()

    log_result({
        "args": {
            "model_name": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_len": MAX_LEN,
            "seed": SEED
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    },log_path)
    print(f"\n训练日志已保存至: {SAVE_DIR}")


if __name__ == "__main__":
    train_main()
    


# import torch
# from torch import nn
# import numpy as np
# import random
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from transformers import AutoTokenizer
# from dataloader import get_data_loader
# from model import Classifier
# from tqdm import tqdm
# from sklearn.metrics import precision_score, recall_score, f1_score
# from pathlib import Path
# import datetime
# import json

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'设备：{device}')

# def set_seed(seed=42):
#     random.seed(seed)                      # Python 随机数
#     np.random.seed(seed)                   # NumPy 随机数
#     torch.manual_seed(seed)                # PyTorch CPU
#     torch.cuda.manual_seed(seed)           # PyTorch GPU
#     torch.cuda.manual_seed_all(seed)       # 多卡 GPU
#     # torch.backends.cudnn.deterministic = True   # 固定算法
#     # torch.backends.cudnn.benchmark = False      # 禁用性能自动优化

# # 参数设置
# MODELS=("bert-base-uncased","open_llama_700M","roberta-base","prajjwal1/bert-tiny","Qwen/Qwen2.5-0.5B",'bert-large-uncased')
# MODEL_NAME ="Qwen/Qwen2.5-1.5B" #
# MODEL_PATH_DIR=Path(os.getcwd())/'result'
# os.path.exists(MODEL_PATH_DIR) or os.makedirs(MODEL_PATH_DIR)
# MODEL_PATH = "best_model.pth"       # 训练阶段保存的模型路径
# EPOCHS = 1
# BATCH_SIZE = 8  # 体积大，减小 batch size 防止 OOM
# LR = 2e-5
# MAX_LEN = 512
# SEED=42
# weight_decay=0.01
# set_seed(SEED)

# # 加载 Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # 编码函数
# def encode_batch(batch):
#     texts, labels = batch
#     encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
#     return encodings['input_ids'], encodings['attention_mask'], labels.detach().clone()

# # 初始化模型
# model = Classifier(model_name=MODEL_NAME)
# model = model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR,weight_decay=weight_decay)
# loss_fn = nn.BCEWithLogitsLoss()

# # 训练函数
# def train():
#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss = 0
#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
#             input_ids, attention_mask, labels = encode_batch(batch)
#             input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)

#             optimizer.zero_grad()
#             logits = model(input_ids, attention_mask).squeeze(-1)
#             loss = loss_fn(logits, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
#     os.path.exists(timestamp) or os.makedirs(timestamp)
#     torch.save(model.state_dict(), Path(timestamp)/MODEL_PATH)

# # 评估函数
# def evaluate():
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating"):
#             input_ids, attention_mask, labels = encode_batch(batch)
#             input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
#             labels = labels.to(device)

#             logits = model(input_ids, attention_mask).squeeze(-1)
#             probs = torch.sigmoid(logits)
#             preds = torch.round(probs)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     precision = precision_score(all_labels, all_preds)
#     recall = recall_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)

#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1-score:  {f1:.4f}")
#     return precision,recall,f1
  
  
# def log_result(result):
#     os.path.exists(timestamp) or os.makedirs(timestamp)
#     with open(f'{timestamp}/train.json','w') as logf:
#         json.dump(result,logf,indent=1)
        
# if __name__ == "__main__":
#     train_loader, test_loader = get_data_loader(BATCH_SIZE)
#     train()
#     precision,recall,f1=evaluate()
#     train_log={
#         'args':{
#             'model_name':MODEL_NAME,
#             'epoch':EPOCHS,
#             'batch_size':BATCH_SIZE,
#             'lr':LR,
#             'max_len':MAX_LEN,
#             'seed':SEED
#         },
#         'result':[precision,recall,f1]
#     }
#     log_result(train_log)