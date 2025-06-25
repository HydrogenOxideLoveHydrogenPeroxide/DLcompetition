import pandas as pd
import re
from pathlib import Path

train_data_path = Path("data/train/train_cleaned.jsonl")
test_data_path = Path("data/test_521/test.jsonl")

# 使用 lines=True 参数读取 JSONL 文件
try:
    train_data_df = pd.read_json(train_data_path, lines=True)
except ValueError as e:
    print(f"Error reading JSONL file: {e}")
    # 如果文件路径或格式有问题，可以尝试以下方法
    # 逐行读取 JSONL 文件
    data = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    train_data_df = pd.DataFrame(data)
def check_spacing(text):
    if re.search(r' (\.)', text):
        return 0
    else:
        return 1

train_data_df['judge'] = train_data_df['text'].apply(check_spacing)

train_data_df.head()




from sklearn.metrics import precision_score, recall_score, f1_score

y_true = train_data_df['label']
y_pred = train_data_df['judge']

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

test_data_df = pd.read_json(test_data_path, lines=True)
test_data_df['judge'] = test_data_df['text'].apply(check_spacing)
test_data_df['judge'].to_csv('submit.txt', index=False, header=False)