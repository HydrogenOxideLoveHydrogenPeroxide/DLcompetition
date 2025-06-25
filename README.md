# 使用不同体量的大模型框架进行尝试
"prajjwal1/bert-tiny"
"bert-base-uncased"
"roberta-base"
"Qwen/Qwen2.5-0.5B"
"Qwen/Qwen2.5-1.5B"

# 正则化
AdamW，它本身就带有权重衰减参数 weight_decay，这就是 L2 正则化。

在初始化 optimizer 时加入 weight_decay 参数

# Qlora
QLoRA（PEFT + 4bit 量化）训练 