import torch
import pandas as pb
from transformers import *

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

model = AutoModelWithLMHead.from_pretrained("bert-large-uncased")

csvdata = pb.read_csv('file.csv', encoding='utf-8')
 
