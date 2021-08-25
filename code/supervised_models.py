import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers.optimization import AdamW
from transformers import  BertTokenizer, BertModel


class LogisticRegression(torch.nn.Module):
    def __init__(self, x_size, n_classes): 
        super(LogisticRegression, self).__init__()             
        self.linear = nn.Linear(x_size, n_classes)
        
    def forward(self, x):
        x = self.linear(x.float())
        x = F.softmax(x, dim=0)
        return x

class BasicLSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, x_size, n_classes, embedding_dim=300): 
        super(BasicLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.n_classes = n_classes
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        out, (ht, ct) = self.lstm(x)        
        x = self.linear(ht[-1])
        return F.softmax(x, dim=0)

class BiLSTM_model(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_classes, hidden_size=64,
                 pretrained_embeddings=None, padding_idx=0, max_length = 110):
        super(BiLSTM_model, self).__init__()

        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=self.embedding_size,num_embeddings=self.num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=self.embedding_size, num_embeddings=self.num_embeddings,
                                    padding_idx=padding_idx, _weight=pretrained_embeddings)
            self.emb.weight.requires_grad = False
        self.dropout = nn.Dropout(0.3)            
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, dropout = 0.5,bidirectional = True)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.linear = nn.Linear(self.hidden_size*2*self.max_length, num_classes) 
            
    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        out, (ht, ct) = self.lstm(x)
        attn = self.attn(out)
        attn_weights = F.softmax(torch.tanh(attn), dim=1)
        attn_applied = torch.bmm(attn_weights, out)
        attn_applied = attn_applied.flatten(1) 
        return F.softmax(self.linear(attn_applied), dim = 0)

