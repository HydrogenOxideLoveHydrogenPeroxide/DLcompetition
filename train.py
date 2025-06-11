# 模型训练部分
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from dataloader import TextDataset,TrainDataPath,TestDataPath
from torch.utils.data import Dataset, DataLoader
import torch

# 训练参数和配置
batch_size = 8
max_length = 512
num_epochs = 3
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 数据加载
train_dataset = TextDataset(TrainDataPath, tokenizer)
test_dataset = TextDataset(TestDataPath, tokenizer)

# 数据预处理和封装为DataLoader
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=learning_rate,
)

# 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

print("训练完成，模型已保存")