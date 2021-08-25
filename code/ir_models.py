import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers.optimization import AdamW
from transformers import  BertTokenizer, BertModel


class LSTM_QA(nn.Module):
    def __init__(self, vocab_size, hidden_size, x_size, n_classes, embedding_size=300,
                 padding_idx=0, pretrained_embeddings=None): 
        super(LSTM_QA, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=self.embedding_size,num_embeddings=self.vocab_size,
                                    padding_idx=padding_idx)
        else:
            print('Loading pretrained embeddings...')
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=self.embedding_size, num_embeddings=self.vocab_size,
                                    padding_idx=padding_idx, _weight=pretrained_embeddings)
            self.emb.weight.requires_grad = False
        
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, dropout=0.5,bidirectional=True)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(self.hidden_size*2, 64)  
        self.linear1 = nn.Linear(64, self.n_classes)        
        
    def forward(self, x_0, x_1):
        x_0 = self.emb(x_0)
        x_1 = self.emb(x_1)
        out_0, (ht_0, ct_0) = self.lstm(x_0)
        out_1, (ht_1, ct_1) = self.lstm(x_1)        
        x = self.cosine(out_0, out_1)
        x = self.linear(x)
        x = self.linear1(x)
        x = F.sigmoid(x)
        return x

class LSTM_CNN_QA(nn.Module):
    def __init__(self, vocab_size, hidden_size, x_size, n_classes, embedding_size=300,
                 padding_idx=0, pretrained_embeddings=None): 
        super(LSTM_CNN_QA, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=self.embedding_size,num_embeddings=self.vocab_size,
                                    padding_idx=padding_idx)
        else:
            print('Loading pretrained embeddings...')
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=self.embedding_size, num_embeddings=self.vocab_size,
                                    padding_idx=padding_idx, _weight=pretrained_embeddings)
            self.emb.weight.requires_grad = False
        
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, dropout=0.5,bidirectional=True)        
        self.conv = nn.Conv1d(in_channels=2, out_channels=10, kernel_size=3)        
        self.cosine = nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(self.hidden_size*2, 64)  
        self.linear1 = nn.Linear(64, self.n_classes)        
        
    def forward(self, x_0, x_1):
        x_0 = self.emb(x_0)
        x_1 = self.emb(x_1)
        out_0, (ht_0, ct_0) = self.lstm(x_0)
        out_1, (ht_1, ct_1) = self.lstm(x_1) 
        ht_0 = ht_0.transpose(0, 1)
        ht_1 = ht_1.transpose(0, 1)
        ht_0 = self.conv(ht_0)
        ht_1 = self.conv(ht_1)
        x = self.cosine(out_0, out_1)
        x = self.linear(x)
        x = self.linear1(x)
        x = F.sigmoid(x)
        return x

class BERT_QA(nn.Module):
    def __init__(self, num_labels=1, pretrained_model='bert-base-uncased', seq_length=30):
        super(BERT_QA, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.classifier = nn.Linear(config.hidden_size, 256)
        self.classifier_1 = nn.Linear(256, self.num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids_0, attention_mask_0, input_ids_1, attention_mask_1, labels=None, output_hidden_states=True):
        outputs_0 = self.bert(input_ids_0, attention_mask=attention_mask_0, output_hidden_states=output_hidden_states)
        outputs_1 = self.bert(input_ids_1, attention_mask=attention_mask_1, output_hidden_states=output_hidden_states)
        a = torch.cat(outputs_0[2][:4], dim=1)
        b = torch.cat(outputs_1[2][:4], dim=1)
        s = self.cosine(a, b)
        x = self.classifier(s)
        x = self.dropout(x)
        x = self.classifier_1(x)
        return F.sigmoid(x)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True