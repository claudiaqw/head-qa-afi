import numpy as np
import pandas as pd
from datasets import load_dataset

from torch.utils.data import Dataset


def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


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

class SequenceVocabulary(object):
    def __init__(self, vocab2index = {}):
        if vocab2index is None:
            vocab2index = {}            
        
        self.vocab2index = vocab2index        
        self.index2vocab = {idx:token for token, idx in self.vocab2index.items()}
              
        self.mask = "<MASK>"
        self.unk = "<UNK>"
        self.begin_seq = "<BEGIN_OF_SEQUENCE>" 
        self.end_seq = "<END_OF_SEQUENCE>"
        self.unk_index = self.add_token(self.unk)
        self.mask_index = self.add_token(self.mask)
        self.begin_seq_index = self.add_token(self.begin_seq)
        self.end_seq_index = self.add_token(self.end_seq)
        
    def add_token(self, token):
        if token in self.vocab2index:
            index = self.vocab2index[token]
        else:
            index = len(self.vocab2index)
            self.vocab2index[token] = index
            self.index2vocab[index] = token
        return index

    def lookup_token(self, token):
            return self.vocab2index.get(token, self.unk_index)

    def lookup_index(self, index):
        if index not in self.index2vocab:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.index2vocab[index]
    
    def __len__(self):
        return len(self.vocab2index)

class Vectorizer(object):
    def __init__(self, vocab, labels, vocab2index, label2index):
        self.sentence_vocab = vocab
        self.label_vocab = labels
        self.vocab2index = vocab2index
        self.label2index = label2index

    def vectorize(self, x, N, padding_start = False):
        enc = np.zeros(N, dtype=np.int32)
        enc1 = np.array([self.vocab2index.get(w, self.vocab2index["UNK"]) for w in x]) #value if w is in voca2index, else vocab2index["UNK"] 
        l = min(N, len(enc1))
        
        if padding_start:
            enc[:l] = enc1[:l]
        else:
            enc[N-l:] = enc1[:l]
        return enc

    @classmethod
    def vectorize_training(cls, array):        
        vocab = Vocabulary(vocab2index = {'UNK': 0})
        label_vocab = Vocabulary(vocab2index = {}, add_unk = False)        
        
        for item in array:
            sample_tok = item['sample_tok']
            for tok in sample_tok:
                vocab.add_token(tok)
            label = item['label']            
            label_vocab.add_token(label)
        return cls(vocab, label_vocab, vocab.vocab2index, label_vocab.vocab2index)

class HeadQA(Dataset):
    def __init__(self, instances, vectorizer, language='es', max_length=150, right_padding = False):
        self.instances = instances
        self.data= load_dataset('head_qa', language)
        self.vectorizer = vectorizer
        self.max_length = max_length
        self.right_padding = right_padding

    def __getitem__(self, index):
        item = self.instances[index]
        sample_tok = item['sample_tok']
        label = item['label']
        x = self.vectorizer.vectorize(sample_tok, self.max_length, self.right_padding)
        y = self.vectorizer.label_vocab.lookup_token(label)
        return x, y

    def __len__(self):
        return len(self.instances)