# 文本预处理函数
def text_to_tensor(text, max_length=512, tokenizer=None):
    # 这里使用BERT tokenizer进行文本预处理
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs