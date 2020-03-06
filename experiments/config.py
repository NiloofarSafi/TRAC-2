import torch

bert_model = 'bert-base-uncased'

label2idx = {'NAG': 0, 'OAG': 1, 'CAG': 2}
idx2label = {0: 'NAG', 1: 'OAG', 2: 'CAG'}

gen2idx = {'NGEN': 0, 'GEN': 1}
idx2gen = {0: 'NGEN', 1: 'GEN'}

VOCAB_PATH = '../torch_moji/vocabulary.json'
maxlen = 300

batch_size = 8
btch_size = 32

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')