import numpy as np
import pandas as pd
from datasets import load_dataset

import torch
from torch.utils.data import Dataset

import spacy
nlp = spacy.load("es_core_news_sm")

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
    def __init__(self, instances, vectorizer, language='es', max_length=20, right_padding = False):
        self.instances = instances
        # self.data= load_dataset('head_qa', language)
        self.vectorizer = vectorizer
        self.max_length = max_length
        self.right_padding = right_padding

    def __getitem__(self, index):
        item = self.instances[index]
        sample_tok = item['sample_tok']
        label = item['label']
        x, y = self.vectorize(sample_tok, label)
        return x, y

    def __len__(self):
        return len(self.instances)

    # recibe una pregunta raw y retorna una lista de 5 (o 5) elementos (x, y)
    # con la pregunta codificada (preg + [SEP] + ans) y el label correspondiente.
    def encode(self, sample):
        qtext, answers = sample['qtext'], sample['answers']
        q = nlp(qtext)
        tok_qtext = [token.text for token in q]
        right_answer = sample['ra']
        X, Y = [], []
        for answer in answers:
            aid, atext = answer['aid'], answer['atext']
            a = nlp(atext)
            tok_atext = [token.text for token in a]
            instance_x = tok_qtext + ['SEP'] + tok_atext
            instance_y = 1 if right_answer == aid else 0
            x, y = self.vectorize(instance_x, instance_y)
            X.append(x)
            Y.append(y)
        return torch.Tensor(X), torch.Tensor(Y)

    def vectorize(self, instance, label):
        x = torch.Tensor(self.vectorizer.vectorize(
            instance, self.max_length, self.right_padding))
        y = torch.Tensor([self.vectorizer.label_vocab.lookup_token(label)])
        return x, y


def parse_dataset(dataset):
    train = []
    for sample in dataset:
        qtext, answers = sample['qtext'], sample['answers']
        q = nlp(qtext)
        tok_qtext = [token.text for token in q]
        right_answer = sample['ra']
        for answer in answers:
            aid, atext = answer['aid'], answer['atext']
            a = nlp(atext)
            tok_atext = [token.text for token in a]
            instance_x = tok_qtext + ['SEP'] + tok_atext
            instance_y = 1 if right_answer == aid else 0
            training_sample = {}
            training_sample['question'] = qtext
            training_sample['answer'] = atext
            training_sample['label'] = instance_y
            training_sample['sample_tok'] = instance_x
            training_sample['category'] = sample['category']
            train.append(training_sample)
    return train











