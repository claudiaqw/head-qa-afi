import torch
import torch.nn.functional as F

import numpy as np

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



