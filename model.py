import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    AutoModelForCausalLM, AutoTokenizer, AutoModel
)
from pathlib import Path
import os

class QwenClassifier(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", dropout=0.3):
        super(QwenClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: (batch, seq_len, hidden)
        cls_embedding = hidden_states[:, 0, :]  # 取第一个 token（类似 [CLS]）
        out = self.dropout(cls_embedding)
        logits = self.classifier(out)  # shape: (batch, 1)
        return logits