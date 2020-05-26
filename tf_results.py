
import torch
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("./model_save")
        self.tokenizer = AutoTokenizer.from_pretrained("./model_save")

    def getAll(self, data):
        inputs = self.tokenizer.encode(data, return_tensors="pt")
        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs)
        import wordintmap
        ogmap = wordintmap.getData()
        inv_map = {v: k for k, v in ogmap.items()}
        data =predictions.tolist()
        text = inv_map[data]
        print(data)
        return text
