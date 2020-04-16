import torch

bert_model = 'bert-base-uncased'

label2idx = {'NAG': 0, 'OAG': 1, 'CAG': 2}
idx2label = {0: 'NAG', 1: 'OAG', 2: 'CAG'}

gen2idx = {'NGEN': 0, 'GEN': 1}
idx2gen = {0: 'NGEN', 1: 'GEN'}

batch_size = 26

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
