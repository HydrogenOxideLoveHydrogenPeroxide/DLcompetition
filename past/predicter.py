import json
import torch
from transformers import AutoTokenizer
from model import Classifier  # 根据你使用的模型类名导入
from pathlib import Path
from tqdm import tqdm
from train import MODEL_NAME,BATCH_SIZE,MAX_LEN,MODEL_PATH  # 导入训练时的参数

# 配置

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
DataDir = Path.cwd() / 'data'
TestDataPath = DataDir / 'test_521' / 'test.jsonl'
OutputPath = "submit.txt"

# 加载模型 & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = Classifier(model_name=MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 加载测试数据（无标签）
def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip())['text'] for line in f]
    return lines

test_texts = load_test_data(TestDataPath)

# 分批预测
def predict(texts):
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
            batch_texts = texts[i:i + BATCH_SIZE]
            encodings = tokenizer(batch_texts, truncation=True, padding=True,
                                  max_length=MAX_LEN, return_tensors="pt")
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = torch.round(probs).long().tolist()
            all_preds.extend(preds)
    return all_preds

# 执行预测
predictions = predict(test_texts)

# 保存预测结果
with open(OutputPath, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(f"{json.dumps(pred)}\n")  # 保证可用 json 解析，每行一个数字
