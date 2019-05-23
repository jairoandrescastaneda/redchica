import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #256X256
        self.convolucionalOne = nn.Conv2d(3,10,5)
        #252X252
        self.pool = nn.MaxPool2d(2,2)
        #126X126
        self.convolucionalTwo = nn.Conv2d(10,20,5)
        #122X122
        #61X61
        self.funcionLinealOne = nn.Linear(20*61*61,450)
        self.funcionLinealTwo = nn.Linear(450,27)

    def forward(self,x):
        x = self.convolucionalOne(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.convolucionalTwo(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,20*61*61)
        x = self.funcionLinealOne(x)
        x = F.relu(x)
        x = self.funcionLinealTwo(x)

        return F.log_softmax(x,dim=1)
    

        

