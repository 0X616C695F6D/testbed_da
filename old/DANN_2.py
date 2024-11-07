# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:32:12 2024

@author: Jielun
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:18:49 2024

@author: Jielun
"""

import h5py
import numpy as np

# Define the path to your HDF5 file
file_path = 'GOLD_XYZ_OSC.0001_1024.hdf5'

# Define the actual class names and map them to their indices
classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
           'FM', 'GMSK', 'OQPSK']

# Define the class names of interest
# classes_of_interest_names = ['16QAM', '64QAM', '8ASK', 'BPSK', 'QPSK', '8PSK','GMSK']
# classes_of_interest_names = ['256QAM', '64QAM', 'OQPSK', 'BPSK', 'QPSK', 'FM','GMSK']
# classes_of_interest_names = ['8ASK', 'BPSK', '32QAM', '16QAM', 'AM-DSB-SC','AM-SSB-SC']
# classes_of_interest_names = ['AM-DSB-WC', 'AM-DSB-SC','FM', 'GMSK', 'OQPSK','64QAM','128QAM']
# classes_of_interest_names = ['AM-DSB-WC', 'AM-DSB-SC','FM', 'GMSK', 'OQPSK','128QAM']
# classes_of_interest_names = ['8ASK', 'BPSK','8PSK', '16PSK','32QAM', '64QAM','OQPSK']
# classes_of_interest_names = ['16QAM', '64QAM', '8ASK', 'BPSK', 'QPSK', '8PSK','GMSK']
classes_of_interest_names = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK']

num_epochs = 10


#%%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Function



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

#%% CNN

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
#         self.pool = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         return x

# class LabelPredictor(nn.Module):
#     def __init__(self):
#         super(LabelPredictor, self).__init__()
#         self.fc1 = nn.Linear(32 * 256, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(-1, 32 * 256)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class DomainClassifier(nn.Module):
#     def __init__(self):
#         super(DomainClassifier, self).__init__()
#         self.fc1 = nn.Linear(32 * 256, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x, alpha):
#         x = grad_reverse(x, alpha)
#         x = x.view(-1, 32 * 256)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


#%% CLDNN


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        x = self.conv1(x)  # Output shape: (batch_size, 64, 1017)
        x = self.pool(x)   # Output shape: (batch_size, 64, 508)
        x = x.permute(0, 2, 1)  # Output shape: (batch_size, 508, 64)
        x, _ = self.lstm1(x)  # Output shape: (batch_size, 508, 64)
        return x

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(508 * 64, 128)
        self.fc2 = nn.Linear(128, 24)

    def forward(self, x):
        x, _ = self.lstm2(x)  # Output shape: (batch_size, 508, 64)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0), -1)  # Output shape: (batch_size, 508 * 64)
        x = torch.relu(self.fc1(x))  # Output shape: (batch_size, 128)
        x = self.fc2(x)  # Output shape: (batch_size, 10)
        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(508 * 64, 100)
        self.fc2 = nn.Linear(100, 2)  # Assuming 2 domains

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = x.contiguous().view(x.size(0), -1)  # Output shape: (batch_size, 508 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output shape: (batch_size, 2)
        return x
    
    
    
    
#%%


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor()
        self.domain_classifier = DomainClassifier()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(features, alpha)
        return class_output, domain_output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DANN().to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%%

# Load the filtered data
X_selected = np.load('S_X_selected.npy')
Y_selected_labels = np.load('S_Y_selected_labels.npy')

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_selected, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_selected_labels, dtype=torch.long).to(device)
# Reshape X_tensor to have a channel dimension
X_tensor = X_tensor.permute(0, 2, 1)  # Shape: (num_samples, 1, 2, 128)


# Combine input and target tensors into a dataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 80
S_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
S_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#% Load T test
# Load the filtered data
X_selected = np.load('T_X_selected.npy')
Y_selected_labels = np.load('T_Y_selected_labels.npy')

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_selected, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_selected_labels, dtype=torch.long).to(device)
# Reshape X_tensor to have a channel dimension
X_tensor = X_tensor.permute(0, 2, 1)  # Shape: (num_samples, 1, 2, 128)

# Combine input and target tensors into a dataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 80
T_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
T_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#%% PER-CLASS PERFORAMNCE

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt



# def train_dann(model, S_train_loader, T_train_loader, optimizer, num_epochs=num_epochs, alpha=0.1):
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss, total_domain_loss, total_class_loss = 0, 0, 0
#         len_dataloader = min(len(S_train_loader), len(T_train_loader))
#         data_source_iter = iter(S_train_loader)
#         data_target_iter = iter(T_train_loader)

#         for i in range(len_dataloader):
#             p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
#             alpha = 2. / (1. + np.exp(-10 * p)) - 1

#             # Training model using source data
#             s_data, s_label = next(data_source_iter)
#             s_data, s_label = s_data.to(device), s_label.to(device)
#             model.zero_grad()
#             class_output, domain_output = model(s_data, alpha)
#             err_s_label = criterion_class(class_output, s_label)
#             err_s_domain = criterion_domain(domain_output, torch.zeros(s_data.size(0), dtype=torch.long).to(device))

#             # Training model using target data
#             t_data, _ = next(data_target_iter)
#             t_data = t_data.to(device)
#             _, domain_output = model(t_data, alpha)
#             err_t_domain = criterion_domain(domain_output, torch.ones(t_data.size(0), dtype=torch.long).to(device))

#             # Combining the losses
#             loss = err_s_label + err_s_domain + err_t_domain
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_domain_loss += err_s_domain.item() + err_t_domain.item()
#             total_class_loss += err_s_label.item()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len_dataloader:.4f}, Domain Loss: {total_domain_loss/len_dataloader:.4f}, Class Loss: {total_class_loss/len_dataloader:.4f}')

# def evaluate_and_plot_confusion_matrix(model, loader, title, num_classes):
#     model.eval()
#     true_labels = []
#     predictions = []
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             class_outputs, _ = model(inputs, alpha=0)  # Alpha set to 0 during inference
#             _, preds = torch.max(class_outputs, 1)
#             true_labels.extend(labels.cpu().numpy())
#             predictions.extend(preds.cpu().numpy())

#     # Calculate overall metrics
#     accuracy = accuracy_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions, average='macro')
#     recall = recall_score(true_labels, predictions, average='macro')
#     f1 = f1_score(true_labels, predictions, average='macro')
    
#     # Compute the confusion matrix
#     conf_mat = confusion_matrix(true_labels, predictions)
    
#     # Plot the confusion matrix
#     plt.figure(figsize=(8,6), dpi=300)
#     sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes_of_interest_names,
#                 yticklabels=classes_of_interest_names)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f'Confusion Matrix - {title}')
#     plt.show()

#     # Calculate per-class accuracy
#     per_class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
    
#     return accuracy, precision, recall, f1, per_class_accuracy

# # Number of repetitions
# num_repeats = 10
# source_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'per_class_accuracy': []}
# target_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'per_class_accuracy': []}

# for _ in range(num_repeats):
#     # Reinitialize the model and optimizer for each repetition
#     model = DANN().to(device)  # Replace model_class with the actual model class
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Replace with the actual optimizer and parameters

#     # Train the model
#     train_dann(model, S_train_loader, T_train_loader, optimizer)

#     # Evaluate on source domain
#     accuracy_s, pr_s, re_s, f1_s, per_class_accuracy_s = evaluate_and_plot_confusion_matrix(model, S_val_loader, "Source Domain", 10)
#     source_metrics['accuracy'].append(accuracy_s)
#     source_metrics['precision'].append(pr_s)
#     source_metrics['recall'].append(re_s)
#     source_metrics['f1'].append(f1_s)
#     source_metrics['per_class_accuracy'].append(per_class_accuracy_s)

#     # Evaluate on target domain
#     accuracy_t, pr_t, re_t, f1_t, per_class_accuracy_t = evaluate_and_plot_confusion_matrix(model, T_val_loader, "Target Domain", 10)
#     print(f'{accuracy_t*100:.2f}\n\n')
#     target_metrics['accuracy'].append(accuracy_t)
#     target_metrics['precision'].append(pr_t)
#     target_metrics['recall'].append(re_t)
#     target_metrics['f1'].append(f1_t)
#     target_metrics['per_class_accuracy'].append(per_class_accuracy_t)

# # Calculate and print average metrics
# avg_source_metrics = {metric: np.mean(values) for metric, values in source_metrics.items() if metric != 'per_class_accuracy'}
# avg_target_metrics = {metric: np.mean(values) for metric, values in target_metrics.items() if metric != 'per_class_accuracy'}

# print(f"Average metrics on Target Domain over {num_repeats} runs:")
# for metric, value in avg_target_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")

# avg_target_per_class_accuracy = np.mean(np.array(target_metrics['per_class_accuracy']), axis=0)
# print("Average per-class accuracy on Target Domain:")
# for class_name, acc in zip(classes_of_interest_names, avg_target_per_class_accuracy):
#     print(f"{class_name}: {acc*100:.2f}")





#%% MEAN PERFORAMNCE


# def train_dann(model, S_train_loader, T_train_loader, optimizer, num_epochs=20, alpha=0.1):
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss, total_domain_loss, total_class_loss = 0, 0, 0
#         len_dataloader = min(len(S_train_loader), len(T_train_loader))
#         data_source_iter = iter(S_train_loader)
#         data_target_iter = iter(T_train_loader)

#         for i in range(len_dataloader):
#             p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
#             alpha = 2. / (1. + np.exp(-10 * p)) - 1

#             # Training model using source data
#             s_data, s_label = next(data_source_iter)
#             s_data, s_label = s_data.to(device), s_label.to(device)
#             model.zero_grad()
#             class_output, domain_output = model(s_data, alpha)
#             err_s_label = criterion_class(class_output, s_label)
#             err_s_domain = criterion_domain(domain_output, torch.zeros(s_data.size(0), dtype=torch.long).to(device))

#             # Training model using target data
#             t_data, _ = next(data_target_iter)
#             t_data = t_data.to(device)
#             _, domain_output = model(t_data, alpha)
#             err_t_domain = criterion_domain(domain_output, torch.ones(t_data.size(0), dtype=torch.long).to(device))

#             # Combining the losses
#             loss = err_s_label + err_s_domain + err_t_domain
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_domain_loss += err_s_domain.item() + err_t_domain.item()
#             total_class_loss += err_s_label.item()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len_dataloader:.4f}, Domain Loss: {total_domain_loss/len_dataloader:.4f}, Class Loss: {total_class_loss/len_dataloader:.4f}')

# def evaluate_and_plot_confusion_matrix(model, loader, title, num_classes):
#     model.eval()
#     true_labels = []
#     predictions = []
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             class_outputs, _ = model(inputs, alpha=0)  # Alpha set to 0 during inference
#             _, preds = torch.max(class_outputs, 1)
#             true_labels.extend(labels.cpu().numpy())
#             predictions.extend(preds.cpu().numpy())

#     # Calculate accuracy
#     accuracy = accuracy_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions, average='macro')
#     recall = recall_score(true_labels, predictions, average='macro')
#     f1 = f1_score(true_labels, predictions, average='macro')
    
#     # Compute the confusion matrix
#     conf_mat = confusion_matrix(true_labels, predictions)
    
#     # Plot the confusion matrix
#     plt.figure(figsize=(8,6), dpi=300)
#     sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes_of_interest_names,
#                 yticklabels=classes_of_interest_names)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f'Confusion Matrix - {title}')
#     plt.show()

#     return accuracy, precision, recall, f1

# # Number of repetitions
# num_repeats = 10
# source_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
# target_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

# for _ in range(num_repeats):
#     # Reinitialize the model and optimizer for each repetition
#     model = DANN().to(device)  # Replace model_class with the actual model class
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Replace with the actual optimizer and parameters

#     # Train the model
#     train_dann(model, S_train_loader, T_train_loader, optimizer)

#     # Evaluate on source domain
#     accuracy_s, pr_s, re_s, f1_s = evaluate_and_plot_confusion_matrix(model, S_val_loader, "Source Domain", 10)
#     source_metrics['accuracy'].append(accuracy_s)
#     source_metrics['precision'].append(pr_s)
#     source_metrics['recall'].append(re_s)
#     source_metrics['f1'].append(f1_s)

#     # Evaluate on target domain
#     accuracy_t, pr_t, re_t, f1_t = evaluate_and_plot_confusion_matrix(model, T_val_loader, "Target Domain", 10)
#     target_metrics['accuracy'].append(accuracy_t)
#     target_metrics['precision'].append(pr_t)
#     target_metrics['recall'].append(re_t)
#     target_metrics['f1'].append(f1_t)

# # Calculate and print average metrics
# avg_source_metrics = {metric: np.mean(values) for metric, values in source_metrics.items()}
# avg_target_metrics = {metric: np.mean(values) for metric, values in target_metrics.items()}

# print(f"Average metrics on Source Domain over {num_repeats} runs:")
# for metric, value in avg_source_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")

# print(f"Average metrics on Target Domain over {num_repeats} runs:")
# for metric, value in avg_target_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")








#%% ORIGINAL

def train_dann(model, S_train_loader, T_train_loader, num_epochs=num_epochs, alpha=0.1):
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_domain_loss, total_class_loss = 0, 0, 0
        len_dataloader = min(len(S_train_loader), len(T_train_loader))
        data_source_iter = iter(S_train_loader)
        data_target_iter = iter(T_train_loader)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Training model using source data
            s_data, s_label = next(data_source_iter)
            s_data, s_label = s_data.to(device), s_label.to(device)
            model.zero_grad()
            class_output, domain_output = model(s_data, alpha)
            err_s_label = criterion_class(class_output, s_label)
            err_s_domain = criterion_domain(domain_output, torch.zeros(s_data.size(0), dtype=torch.long).to(device))

            # Training model using target data
            t_data, _ = next(data_target_iter)
            t_data = t_data.to(device)
            _, domain_output = model(t_data, alpha)
            err_t_domain = criterion_domain(domain_output, torch.ones(t_data.size(0), dtype=torch.long).to(device))

            # Combining the losses
            loss = err_s_label + err_s_domain + err_t_domain
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_domain_loss += err_s_domain.item() + err_t_domain.item()
            total_class_loss += err_s_label.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len_dataloader:.4f}, Domain Loss: {total_domain_loss/len_dataloader:.4f}, Class Loss: {total_class_loss/len_dataloader:.4f}')

# Training the model
train_dann(model, S_train_loader, T_train_loader)

# Evaluation can be similarly adjusted to evaluate both source and target domains


#%% EVA Source
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score

import seaborn as sns

def evaluate_and_plot_confusion_matrix(model, loader, title, num_classes):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_outputs, _ = model(inputs, alpha=0)  # Alpha set to 0 during inference
            _, preds = torch.max(class_outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    
    # Compute the confusion matrix
    conf_mat = confusion_matrix(true_labels, predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8,6),dpi=300)
    sns.heatmap(conf_mat, annot=False, fmt='d', cmap='Blues',
            xticklabels=classes_of_interest_names,
            yticklabels=classes_of_interest_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(fontsize=14,rotation=360)
    plt.xticks(fontsize=14,rotation=45)
    plt.title(f'Confusion Matrix')
    plt.show()

    return accuracy,precision,recall,f1

# Call the evaluation function for both validation sets
accuracy_s,pr_s,re_s,f1_s = evaluate_and_plot_confusion_matrix(model, S_val_loader, "Source Domain", 10)
accuracy_t,pr_t,re_t,f1_t = evaluate_and_plot_confusion_matrix(model, T_val_loader, "Target Domain", 10)
print(f"Accuracy on Source Domain: {accuracy_s*100:.2f} pr {pr_s*100:.2f} re {re_s*100:.2f} f1 {f1_s*100:.2f}")
print(f"Accuracy on Target Domain: {accuracy_t*100:.2f} pr {pr_t*100:.2f} re {re_t*100:.2f} f1 {f1_t*100:.2f}")

