import torch
from torch import nn
from torch.nn import functional as F
from models.attention import *
import models.config as C
from models.attention import *
import sys
sys.path.append('../')
import experiments.config as C
from transformers import *


class BERTRA(nn.Module):

    def __init__(self):
        super(BERTRA, self).__init__()

        # self.rnn_units = 20
        self.embedding_dim = 768

        self.bert = BertModel.from_pretrained(C.bert_model, output_hidden_states=True,
                                                                output_attentions=True)

        self.attention = BaseAttention(self.embedding_dim)

        self.sequential = nn.Sequential(
            nn.Linear(self.embedding_dim , 100),

            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
        )

        self.output1 = nn.Linear(100, 3) #for aggression identification
        self.output2 = nn.Linear(100, 2) #to decide whether the message is gendered or not


    def forward(self, sentences, mask):

        hidden, _ = self.bert(sentences)[-2:]
        sentences = hidden[-1]

        attention_applied, attention_weights = self.attention(sentences, mask.float())

        x = self.sequential(attention_applied)
        out1 = F.softmax(self.output1(x), -1)
        out2 = F.softmax(self.output2(x), -1)


        return {
            'y_pred1': out1,
            'y_pred2': out2,
            'weights': attention_weights
        }




