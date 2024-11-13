import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_covariance(input_data):
    n = input_data.size(0)
    mean = torch.mean(input_data, dim=0, keepdim=True)
    input_data_centered = input_data - mean
    cov = (input_data_centered.t() @ input_data_centered) / (n - 1)
    return cov


def coral(source, target):
    d = source.size(1)
    
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    
    loss = torch.sum((source_cov - target_cov) ** 2)
    loss = loss / (4 * d * d)
    return loss


#%% CLDNN

class CLDNN_G(nn.Module):
    def __init__(self):
        super(CLDNN_G, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=64,
            kernel_size=8, stride=1, padding=0
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(
            input_size=64, hidden_size=64,
            num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=64, hidden_size=64,
            num_layers=1, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_bottleneck = nn.Linear(508*64, 512)
        
    def forward(self, x):
        x = self.conv1(x)  
        x = self.pool(x)   
        x = x.permute(0, 2, 1)  
        x, _ = self.lstm1(x)   
        x = self.dropout1(x)
        x, _ = self.lstm2(x)    
        x = self.dropout2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_bottleneck(x)
        return x  


class CLDNN_C(nn.Module):
    def __init__(self, output_dim):
        super(CLDNN_C, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x