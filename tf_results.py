
import torch
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
       self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
       config = BertConfig.from_pretrained("./model_save", num_labels=13,
            output_attentions=False, output_hidden_states=False,)
       self.model = AutoModelForSequenceClassification.from_config(config)

    def getIntent(self, data):
        inputs2 = self.tokenizer.encode_plus(
                            data,                      # Sentence to encode.
                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                            max_length=64,           # Pad & truncate all sentences.
                            pad_to_max_length=True,
                            return_attention_mask=True,   # Construct attn. masks.
                            return_tensors='pt',     # Return pytorch tensors.
                    )
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
