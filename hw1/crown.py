"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# the data for this task has three columns: x y and class.
# the input of nn will be x and y, and the output will be a binary class.
class Full3Net(torch.nn.Module):
  # assume we have a linear nn here:
    def __init__(self, hid=3):
        super(Full3Net, self).__init__()
        # define the structure of the nn
        # define the first hidden layer: size of in feature is 2 and size of out feature is define by variable hid
        self.hidden1 = nn.Linear(2, hid)
        # define the second hidden layer: size of in feature is hid and size of out feature is hid
        self.hidden2 = nn.Linear(hid, hid)
        # define the third layer: the size of input is hid from layer 2, the size of output is 1
        self.hidden3 = nn.Linear(hid, 1)


    def forward(self, input):
        # assume we are having a linear nn.
        # calculate the linear sum of the weight with the input:
        sum1 = self.hidden1(input)
        # apply the activation function: tanh
        self.hid1 = torch.tanh(sum1)
        # calculate the linear sum of the weight with the first hidden layer output after activation
        sum2 = self.hidden2(self.hid1)
        # apply the activation function: tanh
        self.hid2 = torch.tanh(sum2)
        # compute the sum for the final layer
        out_sum = self.hidden3(self.hid2)
        # apply the activation function: sigmoid
        output = torch.sigmoid(out_sum)
        return output


class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None
        return 0*input[:,0]

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        return 0*input[:,0]
