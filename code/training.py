import torch
import torch.nn.functional as F

def train(model, optimizer, train_dl, test_dl, epochs=100):
    y_trues, y_preds = [], []
    epochs_results = []
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y in train_dl:
            batch = y.shape[0]
            out = model(x.float())  
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
        print("Epoch %s train loss  %.4f val loss %.3f and accuracy %.4f" % (i, train_loss, valid_loss, valid_acc))
    return epochs_results

def validate(model, dataloader):
    model.eval()
    loss, right, total = 0, 0, 0
    y_true, y_preds = [], []
    for x, y in dataloader:
        batch = y.shape[0]
        out = model(x.float())
        loss = F.binary_cross_entropy(out, y.float())
        loss += batch*(loss.item())
        total += batch
        pred = torch.max(out, dim=1)[1]
        y_true.append(y)
        y_preds.append(pred)
        right += (pred == y).float().sum().item()
    return loss/total, right/total, y_true, y_preds

def evaluate(model, instance):
    pass


