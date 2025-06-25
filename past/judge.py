import json

# 文件路径
labeled_path = "data/labeled_output.jsonl"
original_path = "data/test_521/test.jsonl"

# 读取 labeled_output.jsonl 中的所有 text 字段，存入集合以加速查找
with open(labeled_path, "r", encoding="utf-8") as f:
    labeled_texts = {json.loads(line)["text"] for line in f}

print(len(labeled_texts))
# 遍历原始文件，找出 text 不在 labeled_texts 中的行号
non_matching_lines = []
with open(original_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):  # 行号从 1 开始
        data = json.loads(line)
        if data["text"] not in labeled_texts:
            non_matching_lines.append(idx)

# 输出不匹配的行号
print("以下行在原始文件中存在，但不在 labeled_output.jsonl 中：")
print(non_matching_lines)