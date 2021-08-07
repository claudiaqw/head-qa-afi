import torch
import torch.nn.functional as F


def train(model, optimizer, train_dl, test_dl, validate, encoder, epochs=100):
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
        test_acc, points = validate(model, test_dl, encoder)
        #y_trues.append(y_real)
        #y_preds.append(y_pred)
        epochs_results.append([train_loss, points, test_acc])
        print("Epoch %s train loss  %.4f points %.3f and accuracy %.4f" %
              (i, train_loss, points, test_acc))
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

def validate_answer(model, dataloader, encoder, pytorch_model=True):
    def evaluator(model, instance, encoder):
        x, y = encoder(instance)
        #x, y = torch.Tensor(x), torch.Tensor(y)
        y_ = model(x)
        pred = torch.max(y_, dim=0)[1]
        acc = (pred + 1 == y).float()
        points = 3 if acc == 1 else -1
        return acc, points
    
    if pytorch_model:
        model.eval()
    right, score = 0, 0
    for instance in dataloader:
        acc, point = evaluator(model, instance, encoder)
        right += acc
        score += point
    return right/len(), score





