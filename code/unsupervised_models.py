import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers.optimization import AdamW
from transformers import  BertTokenizer, BertModel

class BERTSimilarity(torch.nn.Module):
    def __init__(self, pretrained_model = 'bert-base-uncased'):
        super(BERTSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = self.bert.config
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input_ids_0, attention_mask_0, input_ids_1, attention_mask_1, labels=None, output_hidden_states=True):
        question_outputs = self.bert(input_ids_0, attention_mask=attention_mask_0, output_hidden_states=output_hidden_states)
        answer_outputs = self.bert(input_ids_1, attention_mask=attention_mask_1, output_hidden_states=output_hidden_states)
        quest_last_hidden_state = question_outputs[0] #last_hidden_state
        ans_last_hidden_state = answer_outputs[0]
        batch_size = ans_last_hidden_state.shape[0]
        quest_last_hidden_state = quest_last_hidden_state.view(batch_size, 30*self.config.hidden_size)
        ans_last_hidden_state = ans_last_hidden_state.view(batch_size, 30*self.config.hidden_size)
        sim = self.cosine(quest_last_hidden_state, ans_last_hidden_state)
        return sim
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True