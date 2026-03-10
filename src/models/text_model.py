import torch
import torch.nn as nn
from transformers import BertModel


class TextModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(768,256)

    def forward(self,x):

        # ensure input is long type for BERT
        x = x.long()

        outputs = self.bert(input_ids=x)

        cls_output = outputs.last_hidden_state[:,0,:]

        out = self.fc(cls_output)

        return out