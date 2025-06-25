# import torch
# import torch.nn as nn
# from transformers import AutoModel

# class Classifier(nn.Module):
#     def __init__(self, model_name="prajjwal1/bert-tiny", dropout=0.3):
#         super(Classifier, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(dropout)
#         hidden_size = self.bert.config.hidden_size
#         self.classifier = nn.Linear(hidden_size, 1)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS]
#         dropout_output = self.dropout(pooled_output)
#         logits = self.classifier(dropout_output)
#         return logits
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

def get_peft_model_for_classification(model_name, num_labels=1):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype="auto"
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(base_model, peft_config)
    return model