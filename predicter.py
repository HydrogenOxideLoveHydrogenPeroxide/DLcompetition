import json
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from model import get_peft_model_for_classification  # 你训练时的模型函数
from train import MODEL_NAME, BATCH_SIZE, MAX_LEN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
DataDir = Path.cwd() / 'data'
TestDataPath = DataDir / 'test_521' / 'test.jsonl'
OutputPath = "submit.txt"
MODEL_PATH_DIR=Path('results/20250613_091231')
MODEL_PATH=MODEL_PATH_DIR/"best_model.pth"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型并加载权重
model = get_peft_model_for_classification(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 载入测试文本（无标签）
def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip())['text'] for line in f]
    return lines

test_texts = load_test_data(TestDataPath)

# 批量预测函数
def predict(texts):
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
            batch_texts = texts[i:i + BATCH_SIZE]
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = torch.round(probs).long().tolist()
            all_preds.extend(preds)
    return all_preds

# 执行预测
predictions = predict(test_texts)

# 保存结果
with open(OutputPath, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(f"{json.dumps(pred)}\n")  # 一行一个预测标签，JSON格式

print(f"预测完成，结果已保存到 {OutputPath}")
