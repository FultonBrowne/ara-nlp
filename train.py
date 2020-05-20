import torch

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

import pandas as pb

from transformers import *

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

df = pb.read_csv('data.csv', encoding='utf-8')

print('Number of training sentences: {:,}\n'.format(df.shape[0]))

df.sample(10)

sentences = df.sentence.values

labels = df.label.values

max_len = 0

for sent in sentences:

    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

input_ids = []

attention_masks = []

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)

attention_masks = torch.cat(attention_masks, dim=0)

labels = torch.tensor(labels)

print('Original: ', sentences[0])

print('Token IDs:', input_ids[0])

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))

print('{:>5,} validation samples'.format(val_size))
    
batch_size = 32


train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )