import torch
import torch.nn as nn
from transformers import AutoModel

class Classifier(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-tiny", dropout=0.3):
        super(Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS]
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits
