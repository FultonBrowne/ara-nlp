import torch
import spacy
import transformers
from transformers import *
import os
import numpy as np
class __init__():

    def __init__(self):
#       self.tokenizer = AutoTokenizer.from_pretrained("./model_save")
#       config = BertConfig.from_pretrained("./model_save", num_labels=13,
#            output_attentions=False, output_hidden_states=False,)
#       self.model = AutoModelForSequenceClassification.from_config(config)
#       self.model.eval()
       self.spacymods = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}

       

    def tokenizerfun(self, text):
        return input_ids




#    def getIntent(self, data):
#        input_ids = []
#        attention_masks = []
#        encoded_dict = self.tokenizer.encode_plus(
#                        data,                      # Sentence to encode.
#                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                        max_length = 64,           # Pad & truncate all sentences.
#                        pad_to_max_length = True,
#                        return_attention_mask = True,   # Construct attn. masks.
#                        return_tensors = 'pt',     # Return pytorch tensors.
#                   )
#    
#        # Add the encoded sentence to the list.    
#        input_ids.append(encoded_dict['input_ids'])
#    
#        # And its attention mask (simply differentiates padding from non-padding).
#        attention_masks.append(encoded_dict['attention_mask'])
#
#        # Convert the lists into tensors.
#        input_ids = torch.cat(input_ids, dim=0)
#        attention_masks = torch.cat(attention_masks, dim=0)
#        tokens_tensor = torch.tensor(input_ids)
#        segments_tensors = torch.tensor(attention_masks)
#
#        # Print sentence 0, now as a list of IDs.
#        print('Original: ', data)
#        print('Token IDs:', input_ids[0])
#        outputs = self.model(tokens_tensor, segments_tensors)
#        print(outputs)
#        predictions = torch.argmax(outputs[0])
#        print(predictions)
#        import wordintmap
#        ogmap = wordintmap.getData()
#        inv_map = {v: k for k, v in ogmap.items()}
#        data =predictions.tolist()
#        text = inv_map[data]
#        print(data)
#        return [{'type': 'intent', 'data': text}]
    def getPos(self, data, lang):
        print(data)
        datasets = self.spacymods.get(lang)
        if(datasets == None):
            datasets = self.spacymods.get("en")
        doc = datasets(data)
        print([(w.text, w.pos_) for w in doc])
        return [{"type":w.pos_, "data":w.text} for w in doc]

    def getDpos(self, data, lang):
        print(data)
        datasets = self.spacymods.get(lang)
        if(datasets == None):
            datasets = self.spacymods.get("en")
        doc = datasets(data)
        print([(w.text, w.pos_) for w in doc])
        return [{"type":w.tag_, "data":w.text} for w in doc]
    def lemma(self, data, lang):
        print(data)
        datasets = self.spacymods.get(lang)
        if(datasets == None):
            datasets = self.spacymods.get("en")
        doc = datasets(data)
        print([(w.text, w.pos_) for w in doc])
        return [{"type":w.lemma_, "data":w.text} for w in doc]

    def dep(self, data, lang):
        print(data)
        datasets = self.spacymods.get(lang)
        if(datasets == None):
            datasets = self.spacymods.get("en")
        doc = datasets(data)
        print([(w.text, w.pos_) for w in doc])
        return [{"type":w.dep_, "data":w.text} for w in doc]

    def search(self, data, lang):
        print(data)
        datasets = self.spacymods.get(lang)
        if(datasets == None):
            datasets = self.spacymods.get("en")
        doc = datasets(data)
        print([(w.text, w.pos_) for w in doc])
        return [{"type":{"dep": w.dep_, "pos": w.pos_, "tag": w.tag_, "lemma": w.lemma_}, "data":w.text} for w in doc]


