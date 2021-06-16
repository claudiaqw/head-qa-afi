import numpy as np
import pandas as pd
from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    def __init__(self, vocab2index={}, add_unk={}, unk_token='UNK'):
        self.vocab2index = vocab2index
        self.index2vocab = {idx:token for token, idx in self.vocab2index.items()}
        self.add_unk = add_unk
        self.unk_token = unk_token

    def lookup_token(self, token):
        if self.add_unk:
            return self.vocab2index.get(token, self.unk_index)
        else:
            return self.vocab2index[token]

    def lookup_index(self, index):
        if index not in self.index2vocab:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.index2vocab[index]
    
    def __len__(self):
        return len(self.vocab2index)

    def add_token(self, token):
        if token in self.vocab2index:
            index = self.vocab2index[token]
        else:
            index = len(self.vocab2index)
            self.vocab2index[token] = index
            self.index2vocab[index] = token
        return index

class Vectorizer(object):
    def __init__(self, vocab, labels, vocab2index, label2index, tokenizer):
        self.sentence_vocab = vocab
        self.label_vocab = labels
        self.vocab2index = vocab2index
        self.label2index = label2index
        self.tokenizer = tokenizer

    def vectorize(self, s, N):
        padding_start = False
        x = self.tokenizer(s)
        enc = np.zeros(N, dtype=np.int32)
        enc1 = np.array([self.vocab2index.get(w, self.vocab2index["UNK"]) for w in x]) #value if w is in voca2index, else vocab2index["UNK"] 
        l = min(N, len(enc1))
        
        if padding_start:
            enc[:l] = enc1[:l]
        else:
            enc[N-l:] = enc1[:l]
        return enc

class HeadQA(Dataset):
    def __init__(self, data: dict, category):
        self.data_es = load_dataset('head_qa', 'es')
        self.data_en = load_dataset('head_qa', 'en')

