
import torch
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
       self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
       config = BertConfig.from_pretrained("bert-large-uncased", num_labels=13,
            output_attentions=False, output_hidden_states=False,)
       self.model = AutoModelForSequenceClassification.from_config(config)

    def getIntent(self, data):
        inputs = self.tokenizer.encode(data, return_tensors="pt")
        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs)
        import wordintmap
        ogmap = wordintmap.getData()
        inv_map = {v: k for k, v in ogmap.items()}
        data =predictions.tolist()
        text = inv_map[data]
        print(data)
        return [{'type': 'intent', 'data': text}]
