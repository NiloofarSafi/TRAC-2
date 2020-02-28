import torch

bert_model = 'bert-base-uncased'

batch_size = 8
btch_size = 32

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')