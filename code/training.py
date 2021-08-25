import torch
import torch.nn.functional as F
from transformers import BertTokenizer

import numpy as np

def get_optimizer(model, lr=0.01, wd=0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def train(model, optimizer, train_dl, test_dl, validate, epochs=50):
    y_trues, y_preds = [], []
    epochs_results = []
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y in train_dl:
            batch = y.shape[0]
            out = model(x.long())
            loss = F.binary_cross_entropy(out, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += batch*(loss.item())
        train_loss = sum_loss/total
        valid_loss, valid_acc, y_real, y_pred = validate(model, test_dl)
        y_trues.append(y_real)
        y_preds.append(y_pred)
        epochs_results.append([train_loss, valid_loss, valid_acc])
        print("Epoch %s train loss  %.4f valid loss %.3f and accuracy %.4f" %
              (i, train_loss, valid_loss, valid_acc))
    return epochs_results

def train_ir(model, optimizer, train_dl, test_dl, validate, epochs=50):
    y_trues, y_preds = [], []
    epochs_results = []
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x_0, x_1, y in train_dl:
            batch = y.shape[0]
            out = model(x_0.long(), x_1.long())
            loss = F.binary_cross_entropy(out, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += batch*(loss.item())
        train_loss = sum_loss/total
        valid_loss, valid_acc, y_real, y_pred = validate(model, test_dl)
        y_trues.append(y_real)
        y_preds.append(y_pred)
        epochs_results.append([train_loss, valid_loss, valid_acc])
        print("Epoch %s train loss  %.4f valid loss %.3f and accuracy %.4f" %
              (i, train_loss, valid_loss, valid_acc))
    return epochs_results

def validate(model, dataloader):
    model.eval()
    loss, right, total = 0, 0, 0
    y_true, y_preds = [], []
    for x, y in dataloader:
        batch = y.shape[0]
        out = model(x.long())
        loss = F.binary_cross_entropy(out, y.float())
        loss += batch*(loss.item())
        total += batch
        # pred = torch.max(out, dim=1)[1]
        pred = torch.where(out > 0.4, 1, 0)
        y_true.append(y)
        y_preds.append(pred)
        right += (pred == y).float().sum().item()
    return loss/total, right/total, y_true, y_preds

def validate_ir(model, dataloader):
    model.eval()
    loss, right, total = 0, 0, 0
    y_true, y_preds = [], []
    for x_0, x_1, y in dataloader:
        batch = y.shape[0]
        out = model(x_0.long(), x_1.long())
        loss = F.binary_cross_entropy(out, y.float())
        loss += batch*(loss.item())
        total += batch
        # pred = torch.max(out, dim=1)[1]
        pred = torch.where(out > 0.4, 1, 0)
        y_true.append(y)
        y_preds.append(pred)
        right += (pred == y).float().sum().item()
    return loss/total, right/total, y_true, y_preds

def evaluator(model, instance, encoder):
    x, y = encoder(instance)
    y_ = model(x.long())
    pred = torch.max(y_, dim=0)[1]
    real = torch.max(y, dim=0)[1]
    acc = (pred == real).float()
    points = 3 if acc == 1 else -1
    return acc, points

def evaluator_ir(model, instance, encoder):
    x_0, y_0, y = encoder(instance)
    y_ = model(x_0.long(), y_0.long())
    pred = torch.max(y_, dim=0)[1]
    real = torch.max(y, dim=0)[1]
    acc = (pred == real).float()
    points = 3 if acc == 1 else -1
    return acc, points

def evaluate(model, dataloader, encoder, evaluator, pytorch_model=True):
    if pytorch_model:
        model.eval()
    right, score = 0, 0
    for instance in dataloader:
        acc, point = evaluator(model, instance, encoder)
        right += acc
        score += point
    return right/len(dataloader), score

def load_embeddings_from_file(filepath):
    word_to_index, embeddings = {}, []
    with open(filepath, "r", encoding='utf-8') as fp:
        _, emb_size = fp.readline().split()
        index = 0
        for line in fp:            
            line = line.split() # each line: word num1 num2 ...            
            word = line[0]
            if len(line) != int(emb_size) + 1 or word in word_to_index:
                continue
            word_to_index[word] = index
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
            index += 1
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(filepath, words, word_to_idx=None, glove_embeddings=None):
    if word_to_idx is None or glove_embeddings is None:
        word_to_idx, glove_embeddings = load_embeddings_from_file(filepath)
    embedding_size = glove_embeddings.shape[1]
    final_embeddings = np.zeros((len(words), embedding_size))
    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i,:] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size) #si el embedding no esta, se genera a partir de una distribución
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i,:] = embedding_i
    return final_embeddings

# BERT tools
def pad_seq(x, seq_len=110, right_padding = False):
    z = np.zeros(seq_len, dtype=np.int32)
    n = min(seq_len, len(x))
    if right_padding:
        z[:n] = x[0:n]
    else:
        z[(seq_len - n):] = x[0:n]
    return z

def encoder_bert(samples, tokenizer):
    input_ids, labels = [], []
    for item in samples:
        sent = item['question'] +' [SEP] ' + item['answer'] 
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        padded_sent = pad_seq(encoded_sent, seq_len=30)
        input_ids.append(padded_sent)
        labels.append(item['label'])
        
    attention_masks = []
    for sent in input_ids:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
        
    return input_ids, attention_masks, labels

def encoder_bert_ir(samples, tokenizer):
    input_ids_0, input_ids_1, labels = [], [], []
    for item in samples:
        encoded_q = tokenizer.encode(item['question'], add_special_tokens = True)
        encoded_a = tokenizer.encode(item['answer'], add_special_tokens = True)
        padded_q = pad_seq(encoded_q, seq_len=30)
        padded_a = pad_seq(encoded_a, seq_len=30)
        input_ids_0.append(padded_q)
        input_ids_1.append(padded_a)
        labels.append(item['label'])
        
    attention_masks_0, attention_masks_1 = [], []
    for sent in input_ids_0:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_0.append(att_mask)
    
    for sent in input_ids_1:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_1.append(att_mask)
        
    return input_ids_0, attention_masks_0, input_ids_1, attention_masks_1, labels

def encoder_bert_instance(sample, tokenizer=BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', do_lower_case=False)):
    qtext, answers = sample['qtext'], sample['answers']
    right_answer = sample['ra']
    input_ids, labels = [], []
    
    for answer in answers:
        aid, atext = answer['aid'], answer['atext']
        sent = qtext + ' [SEP] ' + atext
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        padded_sent = pad_seq(encoded_sent, seq_len=30)
        input_ids.append(padded_sent)
        instance_y = 1 if right_answer == aid else 0
        labels.append(instance_y)
        
    attention_masks = []
    for sent in input_ids:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
        
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)

def encoder_bert_ir_instance(sample, tokenizer=BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', do_lower_case=False)):    
    qtext, answers = sample['qtext'], sample['answers']    
    encoded_q = tokenizer.encode(qtext, add_special_tokens = True)
    padded_q = pad_seq(encoded_q, seq_len=30)
    right_answer = sample['ra']
    
    input_ids_0, input_ids_1, labels = [], [], []
    right_answer = sample['ra']
    for answer in answers:
        aid, atext = answer['aid'], answer['atext']
        encoded_a = tokenizer.encode(atext, add_special_tokens=True)        
        padded_a = pad_seq(encoded_a, seq_len=30)
        input_ids_0.append(padded_q)
        input_ids_1.append(padded_a)
        instance_y = 1 if right_answer == aid else 0
        labels.append(instance_y)
        
    attention_masks_0, attention_masks_1 = [], []
    for sent in input_ids_0:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_0.append(att_mask)
    
    for sent in input_ids_1:  
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_1.append(att_mask)
        
    return  torch.tensor(input_ids_0),  torch.tensor(attention_masks_0), torch.tensor(input_ids_1),  torch.tensor(attention_masks_1),  torch.tensor(labels)

def evaluator_bert_ir(model, instance, encoder):
    b_input_ids_0, b_input_mask_0, b_input_ids_1, b_input_mask_1, y = encoder(instance)
    y_ = model(input_ids_0=b_input_ids_0, 
              attention_mask_0=b_input_mask_0, 
              input_ids_1=b_input_ids_1,
              attention_mask_1=b_input_mask_1,
              labels=y)
    pred = torch.max(y_, dim=0)[1]
    real = torch.max(y, dim=0)[1]
    acc = (pred == real).float()
    points = 3 if acc == 1 else -1
    return acc, points

def evaluator_bert(model, instance, encoder):
    b_input_ids_0, b_input_mask_0, y = encoder(instance)
    y_ = model(b_input_ids_0, 
              b_input_mask_0,
              labels=y)
    y_ = y_.logits
    pred = torch.max(y_, dim=0)[1]
    real = torch.max(y, dim=0)[1]
    acc = (pred == real).float()
    points = 3 if acc == 1 else -1
    return acc, points

