import torch
import pandas as pb
from transformers import *

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

model = AutoModelWithLMHead.from_pretrained("bert-large-uncased")

df = pb.read_csv('data.csv', encoding='utf-8')

for index, row in df.iterrows():
    tokenized_text = tokenizer.tokenize(row["data"])
    print(row["data"])
    print(row["lable"])
