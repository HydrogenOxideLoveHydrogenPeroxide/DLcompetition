import os 
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from sklearn.model_selection import train_test_split

# ====== 配置 ======
model_id = "Qwen/Qwen2.5-0.5B"
data_path = Path(os.getcwd()) / "data" / "train" / "train.jsonl"
output_dir = "./qwen2_qlora_output"

# ====== 加载并划分数据 ======
def load_and_split_jsonl_dataset(path, test_size=0.1, seed=42):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj['text']
            label = obj['label']
            label_text = "Human" if label == 0 else "AI"
            prompt = f"Text:\n{text}\n\nIs this written by a human or AI? Answer: {label_text}"
            data.append({"text": prompt})

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

datasets = load_and_split_jsonl_dataset(data_path)

# ====== Tokenizer & Model ======
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    use_fast=False
    )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True
)

model = prepare_model_for_kbit_training(model)

# ====== 打印模型模块名，帮助确定 LoRA 注入目标 ======
print("模型部分模块名示例:")
for name, module in model.named_modules():
    if "attn" in name or "mlp" in name or "proj" in name:
        print(name)

# ====== QLoRA 配置 ======
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====== Tokenize ======
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_datasets = datasets.map(tokenize, batched=True)

# ====== Trainer 配置 ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
    label_names=["labels"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

model.config.use_cache = False
trainer.train()

# ====== 保存模型 ======
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ QLoRA 微调完成，模型已保存到", output_dir)