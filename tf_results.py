import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_hub as hub
import bert
import pandas as pd
import numpy as np
from transformers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
def _get_segments(sentences):
    sentences_segments = []
    for sent in sentences:
      temp = []
      i = 0
      for token in sent.split(" "):
        temp.append(i)
        if token == "[SEP]":
          i += 1
      sentences_segments.append(temp)
    return sentences_segments

def _get_inputs(df,_maxlen,tokenizer,use_keras_pad=False):


    maxqnans = np.int((_maxlen-20)/2)
    pattern = '[^\w\s]+|\n' # remove everything including newline (|\n) other than words (\w) or spaces (\s)
    
    sentences = ["[CLS] " + " ".join(tokenizer.tokenize(qn)[:maxqnans]) +" [SEP] " 
              + " ".join(tokenizer.tokenize(ans)[:maxqnans]) +" [SEP] " 
              + " ".join(tokenizer.tokenize(title)[:10]) + " [SEP] "
              + " ".join(tokenizer.tokenize(cat)[:10]) +" [SEP]" 
                for (title,qn,ans,cat) 
                in 
              zip(df['question_title'].str.replace(pattern, '').values.tolist(),
              df['question_body'].str.replace(pattern, '').values.tolist(),
              df['answer'].str.replace(pattern, '').values.tolist(),
              df['category'].str.replace(pattern, '').values.tolist())]
              #train.head()[['question_title','question_body','answer','category']].values.tolist()]
    

    #generate masks
    # bert requires a mask for the words which are padded. 
    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]
    sentences_mask = [[1]*len(sent.split(" "))+[0]*(_maxlen - len(sent.split(" "))) for sent in sentences]
 
    #generate input ids  
    # if less than max length provided then the words are padded
    if use_keras_pad:
      sentences_padded = pad_sequences(sentences.split(" "), dtype=object, maxlen=10, value='[PAD]',padding='post')
    else:
      sentences_padded = [sent + " [PAD]"*(_maxlen-len(sent.split(" "))) if len(sent.split(" "))!=_maxlen else sent for sent in sentences ]

    sentences_converted = [tokenizer.convert_tokens_to_ids(s.split(" ")) for s in sentences_padded]
    
    #generate segments
    # for each separation [SEP], a new segment is converted
    sentences_segment = _get_segments(sentences_padded)

    genLength = set([len(sent.split(" ")) for sent in sentences_padded])

    if _maxlen < 20:
      raise Exception("max length cannot be less than 20")
    elif len(genLength)!=1: 
      print(genLength)
      raise Exception("sentences are not of same size")



    #convert list into tensor integer arrays and return it
    #return sentences_converted,sentences_segment, sentences_mask
    #return [np.asarray(sentences_converted, dtype=np.int32), 
    #        np.asarray(sentences_segment, dtype=np.int32), 
    #        np.asarray(sentences_mask, dtype=np.int32)]
    return [tf.cast(sentences_converted,tf.int32), tf.cast(sentences_segment,tf.int32), tf.cast(sentences_mask,tf.int32)]
def build_model_fullyconnected(MAX_SEQUENCE_LENGTH = 100):
        input_word_ids = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_masks = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
        input_segments = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
        _, sout = bert_layer([input_word_ids, input_masks, input_segments])
        X= Dense(100, activation='relu')(sout) 
        #X= Dense(100, activation='relu')(input_) 
        X = GlobalAveragePooling1D()(X)
        output_= Dense(30, activation='sigmoid', name='output')(X)

        #model = Model(input_,output_)
        model = Model([input_word_ids, input_masks, input_segments],output_)
        print(model.summary())

        return model

def build_model_bertembed(MAX_SEQUENCE_LENGTH = 100):

    input_ = Input(shape = (MAX_SEQUENCE_LENGTH,768), name='bert_enconding')
    X= Dense(100, activation='relu')(input_) 
    X = GlobalAveragePooling1D()(X)
    output_= Dense(30, activation='sigmoid', name='output')(X)
    
    model = Model(input_,output_)
    print(model.summary())
    return model
    msg = "starting tf"
    training = False
class __init__():
    train = pd.read_csv("./kaggle/google_quest/train.csv.zip")
    test = pd.read_csv("./kaggle/google_quest/test.csv")
    sub = pd.read_csv("./kaggle/google_quest/sample_submission.csv")
    def __init__(self):
        bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
        bert_layer = hub.KerasLayer(bert_path,trainable=True)
        vocab_file1 = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        bert_tokenizer_tfhub = bert.bert_tokenization.FullTokenizer(vocab_file1, do_lower_case=True)
        bert_inputs = _get_inputs(df=train.head(),tokenizer=bert_tokenizer_tfhub,_maxlen=100)
        _,Xtr_bert= bert_layer(bert_inputs)
        ytr = np.asarray(train.iloc[:5,-30:])
        model = build_models_fullyconnected()
        model.compile(optimizer = "adam",loss = "binary_crossentropy")
        history = model.fit(bert_inputs,ytr,epochs=1,batch_size = 3)
        model = build_models_bertembed()
        model.compile(optimizer = "adam",loss = "binary_crossentropy")
        history = model.fit(Xtr_bert,ytr,epochs=1,batch_size = 3)
    def getAll():
        print("test")