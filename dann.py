"""
Models with Domain Adversarial Neural Network.
DANN: feature extractor, label predictor, domain classifier
GRL:  gradient reversal layer

FA: extracts features from input data
DC: guess domain of sample, source or target
LP: model prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

#%% GRL
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def grad_reverse(x, alpha=1.0):
    return ReverseLayerF.apply(x, alpha)


#%% DANN
class DANN(nn.Module):
    def __init__(self, FA, LP, DC):
        super(DANN, self).__init__()
        self.feature_extractor = FA()
        self.label_predictor   = LP()
        self.domain_classifier = DC()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(features, alpha)
        return class_output, domain_output


#%% CNN
class CNN_FA(nn.Module):
    def __init__(self):
        super(CNN_FA, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

class CNN_LP(nn.Module):
    def __init__(self, output_dim):
        super(CNN_LP, self).__init__()
        self.fc1 = nn.Linear(32 * 256, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(-1, 32 * 256)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_DC(nn.Module):
    def __init__(self):
        super(CNN_DC, self).__init__()
        self.fc1 = nn.Linear(32 * 256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        x = x.view(-1, 32 * 256)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#%% CLDNN
class CLDNN_FA(nn.Module):
    def __init__(self):
        super(CLDNN_FA, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        return x

class CLDNN_LP(nn.Module):
    def __init__(self, output_dim=24):
        super(CLDNN_LP, self).__init__()
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(508 * 64, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CLDNN_DC(nn.Module):
    def __init__(self):
        super(CLDNN_DC, self).__init__()
        self.fc1 = nn.Linear(508 * 64, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = x.contiguous().view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x