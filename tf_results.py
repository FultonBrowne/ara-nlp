
import torch
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def getAll(self, data):
        label_list = [
            "O",       # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",   # Beginning of a person's name right after another person's name
            "I-PER",   # Person's name
            "B-ORG",   # Beginning of an organisation right after another organisation
            "I-ORG",   # Organisation
            "B-LOC",   # Beginning of a location right after another location
            "I-LOC"    # Location
        ]

        sequence = data

        # Bit of a hack to get the tokens with the special tokens
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence)))
        inputs = self.tokenizer.encode(sequence, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)
        print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])
