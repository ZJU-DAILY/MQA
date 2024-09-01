import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer

k = 1
t = k * 1


class AggregationFunctionModel(nn.Module):

    def __init__(self, modal_num):
        super(AggregationFunctionModel, self).__init__()

        self.omega = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True) for _ in range(modal_num)])

        # self.omega = nn.ParameterList()
        # for i in range(modal_num):
        #     self.omega.append(nn.Parameter(Variable(torch.FloatTensor([0.5]), requires_grad=True)))

    # distance with weight
    def forward(self, d):
        if len(d) == 0:
            return 0
        else:
            return sum(x * self.omega[i] for i, x in enumerate(d))

    def clamp_parameters(self):
        for p in self.parameters():
            p.data.clamp_(0.001, 1.0)
        total = sum(p.data for p in self.omega)
        for p in self.omega:
            p.data = p.data / total


class AggregationFunctionLoss(nn.Module):

    def __init__(self) -> None:
        super(AggregationFunctionLoss, self).__init__()

    def forward(self, phi_positive, phi_negative):
        # for each query, pre-calculate the IP(p, p+) and IP(p, p-)
        loss = (1 / len(phi_positive)) * sum(
            [torch.sum(-torch.log(torch.exp(phi_positive[i] / t) / (torch.exp(phi_positive[i] / t) + torch.exp(phi_negative[i] / t))))
             for i in range(len(phi_positive))])

        print("loss: ", loss)
        return loss


class AggregationFunctionOptimizer(Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        super(AggregationFunctionOptimizer, self).__init__(params, {})

    def step(self, closure=False):
        random_num = 0.4

        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                param.data = param.data - self.lr * random_num * param.grad
