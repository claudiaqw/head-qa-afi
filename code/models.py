import torch
import random
import spacy
nlp = spacy.load("es_core_news_sm")

def random_model(instance):
    n = len(instance['answers'])
    answer = random.randint(0, n-1)
    y_ = torch.tensor(answer)
    return y_.long()

def blind_model(instance, n=0):
    return n

def length_model(instance):
    lengths = []
    for a in instance['answers']:
        atext = answer['atext']
        a = nlp(atext)
        tok_atext = [token.text for token in a]
        lengths.append(len(tok_atext))
    tensor = torch.Tensor(lengths)
    return torch.max(tensor.long(), dim=0)[1]
