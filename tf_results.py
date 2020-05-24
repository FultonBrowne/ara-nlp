
import torch
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("./model_save")

    def getAll(self, data):

        sequence = data

        # Bit of a hack to get the tokens with the special tokens
        inputs = self.tokenizer.encode(sequence, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs)
        data =predictions.tolist()
        print(data)
