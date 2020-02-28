import torch
from torch import nn


class BaseAttention(nn.Module):
    def __init__(self, dimension):
        super(BaseAttention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, mask):

        u_it = self.tanh(self.u(h))
        alpha = torch.exp(torch.matmul(u_it, self.v))
        alpha = mask * alpha + self.epsilon
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        alpha = mask * (alpha / denominator_sum)

        output = h * alpha.unsqueeze(2)
        output = torch.sum(output, dim=1)

        return output, alpha
