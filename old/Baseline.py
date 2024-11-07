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
classes_of_interest_names = ['256QAM', '64QAM', 'OQPSK', 'BPSK', 'QPSK', 'FM','GMSK']
# classes_of_interest_names = ['8ASK', 'BPSK', '32QAM', '16QAM', 'AM-DSB-SC','AM-SSB-SC']
# classes_of_interest_names = ['8ASK', 'BPSK','8PSK', '16PSK','32QAM', '64QAM','OQPSK']
# classes_of_interest_names = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
#             '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
#             '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
#             'FM', 'GMSK', 'OQPSK']

# Map class names to indices
classes_of_interest_indices = [classes.index(name) for name in classes_of_interest_names]


# Define the specific value of Z you are interested in
specific_Z_value = 20  # Replace with the actual Z value

# Load the HDF5 file
with h5py.File(file_path, 'r') as f:
    # Get the data
    X = f['X'][:]
    Y = f['Y'][:]
    Z = f['Z'][:]

# Filter the data to include only the classes of interest and specific Z value
selected_indices = [
    i for i, (label, z_val) in enumerate(zip(np.argmax(Y, axis=1), Z))
    if label in classes_of_interest_indices and z_val == specific_Z_value
]

X_selected = X[selected_indices]
Y_selected = Y[selected_indices]
Z_selected = Z[selected_indices]

# Display shapes of the filtered datasets
print(f'Shape of X_selected: {X_selected.shape}')
print(f'Shape of Y_selected: {Y_selected.shape}')
print(f'Shape of Z_selected: {Z_selected.shape}')

# Optionally, convert one-hot labels to categorical labels for ease of use
templabels = np.argmax(Y_selected, axis=1)
# Get unique labels
unique_labels = np.unique(templabels)

# Map the labels to successive numbers starting from 0
label_map = {label: idx for idx, label in enumerate(unique_labels)}
Y_selected_labels = np.array([label_map[label] for label in templabels])

# Save the filtered data if needed
np.save('S_X_selected.npy', X_selected)
np.save('S_Y_selected_labels.npy', Y_selected_labels)

print("Data filtering complete. Files saved as 'X_selected.npy', 'Y_selected_labels.npy', and 'Z_selected.npy'.")

# Define the specific value of Z you are interested in
specific_Z_value =10# Replace with the actual Z value

# Load the HDF5 file
with h5py.File(file_path, 'r') as f:
    # Get the data
    X = f['X'][:]
    Y = f['Y'][:]
    Z = f['Z'][:]

# Filter the data to include only the classes of interest and specific Z value
selected_indices = [
    i for i, (label, z_val) in enumerate(zip(np.argmax(Y, axis=1), Z))
    if label in classes_of_interest_indices and z_val == specific_Z_value
]

X_selected = X[selected_indices]
Y_selected = Y[selected_indices]
Z_selected = Z[selected_indices]

# Display shapes of the filtered datasets
print(f'Shape of X_selected: {X_selected.shape}')
print(f'Shape of Y_selected: {Y_selected.shape}')
print(f'Shape of Z_selected: {Z_selected.shape}')

# Optionally, convert one-hot labels to categorical labels for ease of use
# Y_selected_labels = np.argmax(Y_selected, axis=1)-len(classes_of_interest_names)
# Optionally, convert one-hot labels to categorical labels for ease of use
templabels = np.argmax(Y_selected, axis=1)
# Get unique labels
unique_labels = np.unique(templabels)

# Map the labels to successive numbers starting from 0
label_map = {label: idx for idx, label in enumerate(unique_labels)}
Y_selected_labels = np.array([label_map[label] for label in templabels])

# Save the filtered data if needed
np.save('T_X_selected.npy', X_selected)
np.save('T_Y_selected_labels.npy', Y_selected_labels)

print("Data filtering complete. Files saved as 'X_selected.npy', 'Y_selected_labels.npy', and 'Z_selected.npy'.")


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt



#%%
# X_selected = np.load('S_X_selected.npy')

# # Select one signal to plot (for example, the first one)
# signal = X_selected[0]

# # Plot the In-phase (I) and Quadrature (Q) components on the same plot
# plt.figure(figsize=(12, 6))

# # Plot I and Q components
# plt.plot(signal[:, 0], label='In-phase (I)')
# plt.plot(signal[:, 1], label='Quadrature (Q)')

# # Adding titles and labels
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])

# plt.show()

# #%%

# X_selected = np.load('T_X_selected.npy')

# # Select one signal to plot (for example, the first one)
# signal = X_selected[0]

# # Plot the In-phase (I) and Quadrature (Q) components on the same plot
# plt.figure(figsize=(12, 6))

# # Plot I and Q components
# plt.plot(signal[:, 0], label='In-phase (I)')
# plt.plot(signal[:, 1], label='Quadrature (Q)')

# # Adding titles and labels
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.show()



#%% CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 256, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 256)  # Flatten the output from conv layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Instantiate the model and move it to CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)


#%% MLP

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.fc1 = nn.Linear(2048, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 10)  # Assuming 10 classes for classification

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Ensure the input is flattened correctly
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
# # Instantiate the model and move it to CUDA device if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN().to(device)

#%% CLDNN

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         # Convolutional layer: from 2 input channels to 64 output channels, kernel size 8
#         self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding=0)
#         # Max pooling layer: kernel size 2, stride 2
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         # LSTM layers
#         self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
#         self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
#         # Dropout layers
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.5)
#         # Fully connected layers
#         self.fc1 = nn.Linear(508 * 64, 128)  # Flattening to 508*64 = 32512 dimensions, then to 128
#         self.fc2 = nn.Linear(128, 24)  # Then to 10 classes

#     def forward(self, x):
#         # Input shape: (batch_size, 2, 1024)
#         x = self.conv1(x)  # Output shape: (batch_size, 64, 1017)
#         x = self.pool(x)   # Output shape: (batch_size, 64, 508)
        
#         # Prepare for LSTM (batch, seq_len, features)
#         x = x.permute(0, 2, 1)  # Output shape: (batch_size, 508, 64)
        
#         # LSTM layers
#         x, _ = self.lstm1(x)  # Output shape: (batch_size, 508, 64)
#         x = self.dropout1(x)
#         x, _ = self.lstm2(x)  # Output shape: (batch_size, 508, 64)
#         x = self.dropout2(x)
        
#         # Flatten the output from LSTM layers
#         x = x.contiguous().view(x.size(0), -1)  # Output shape: (batch_size, 508 * 64)
        
#         # Fully connected layers
#         x = torch.relu(self.fc1(x))  # Output shape: (batch_size, 128)
#         x = self.fc2(x)              # Output shape: (batch_size, 10)
        
#         return x

# # Instantiate the model and move it to CUDA device if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN().to(device)






#%%


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#
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
batch_size = 50
S_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
S_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load T test
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
batch_size = 64
T_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
T_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score
import seaborn as sns


#%% PER-CLASS

num_epochs = 10
num_runs = 10

# Initialize lists to store performance metrics for each run
accuracy_s_list, pr_s_list, re_s_list, f1_s_list = [], [], [], []
accuracy_t_list, pr_t_list, re_t_list, f1_t_list = [], [], [], []
class_accuracies_s = np.zeros((num_runs, len(classes_of_interest_names)))
class_accuracies_t = np.zeros((num_runs, len(classes_of_interest_names)))

def train_and_evaluate_model():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in S_train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(S_train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in S_val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    return model

def evaluate_and_plot_confusion_matrix(model, loader, num_classes):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_outputs = model(inputs)  # Alpha set to 0 during inference
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
    
    # Calculate class-wise accuracy
    class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)

    # Plot the confusion matrix
    plt.figure(figsize=(8,6),dpi=300)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes_of_interest_names,
                yticklabels=classes_of_interest_names)

    plt.yticks(fontsize=14,rotation=360)
    plt.xticks(fontsize=14,rotation=90)
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, f1, class_accuracy

for run in range(num_runs):
    print(f'Run {run+1}/{num_runs}')
    # Reset model and optimizer for each run
    model = CNN().to(device)  # Assuming you have a function to initialize your model
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Assuming you have a function to initialize your optimizer

    trained_model = train_and_evaluate_model()

    # Evaluate on source domain
    accuracy_s, pr_s, re_s, f1_s, class_acc_s = evaluate_and_plot_confusion_matrix(trained_model, S_val_loader, 7)
    accuracy_s_list.append(accuracy_s)
    pr_s_list.append(pr_s)
    re_s_list.append(re_s)
    f1_s_list.append(f1_s)
    class_accuracies_s[run] = class_acc_s
    
    # Evaluate on target domain
    accuracy_t, pr_t, re_t, f1_t, class_acc_t = evaluate_and_plot_confusion_matrix(trained_model, T_val_loader, 7)
    accuracy_t_list.append(accuracy_t)
    pr_t_list.append(pr_t)
    re_t_list.append(re_t)
    f1_t_list.append(f1_t)
    class_accuracies_t[run] = class_acc_t

# Calculate mean and standard deviation of performance metrics
mean_accuracy_s = np.mean(accuracy_s_list)
mean_pr_s = np.mean(pr_s_list)
mean_re_s = np.mean(re_s_list)
mean_f1_s = np.mean(f1_s_list)

mean_accuracy_t = np.mean(accuracy_t_list)
mean_pr_t = np.mean(pr_t_list)
mean_re_t = np.mean(re_t_list)
mean_f1_t = np.mean(f1_t_list)

mean_class_accuracies_s = np.mean(class_accuracies_s, axis=0)
mean_class_accuracies_t = np.mean(class_accuracies_t, axis=0)

print(f"&{mean_accuracy_s*100:.2f}&{mean_pr_s*100:.2f}&{mean_re_s*100:.2f}&{mean_f1_s*100:.2f}")
print(f"&{mean_accuracy_t*100:.2f}&{mean_pr_t*100:.2f}&{mean_re_t*100:.2f}&{mean_f1_t*100:.2f}")

for i, class_name in enumerate(classes_of_interest_names):
    # print(f"Class '{class_name}' Accuracy on Source Domain: {mean_class_accuracies_s[i]*100:.2f}%")
    print(f"{mean_class_accuracies_t[i]*100:.2f}")




#%% MEAN PERFORMANCE


# num_epochs = 10
# num_runs = 20

# # Initialize lists to store performance metrics for each run
# accuracy_s_list, pr_s_list, re_s_list, f1_s_list = [], [], [], []
# accuracy_t_list, pr_t_list, re_t_list, f1_t_list = [], [], [], []

# def train_and_evaluate_model():
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in S_train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(S_train_loader.dataset)
#         print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

#     # Evaluate the model
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in S_val_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     val_accuracy = correct / total
#     print(f'Validation Accuracy: {val_accuracy:.4f}')
    
#     return model

# def evaluate_and_plot_confusion_matrix(model, loader, num_classes):
#     model.eval()
#     true_labels = []
#     predictions = []
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             class_outputs = model(inputs)  # Alpha set to 0 during inference
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
#     plt.figure(figsize=(8,6),dpi=300)
#     sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes_of_interest_names,
#                 yticklabels=classes_of_interest_names)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()

#     return accuracy, precision, recall, f1

# for run in range(num_runs):
#     print(f'Run {run+1}/{num_runs}')
#     # Reset model and optimizer for each run
#     model = CNN().to(device)  # Assuming you have a function to initialize your model
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Assuming you have a function to initialize your optimizer

#     trained_model = train_and_evaluate_model()

#     # Evaluate on source domain
#     accuracy_s, pr_s, re_s, f1_s = evaluate_and_plot_confusion_matrix(trained_model, S_val_loader, 7)
#     accuracy_s_list.append(accuracy_s)
#     pr_s_list.append(pr_s)
#     re_s_list.append(re_s)
#     f1_s_list.append(f1_s)
    
#     # Evaluate on target domain
#     accuracy_t, pr_t, re_t, f1_t = evaluate_and_plot_confusion_matrix(trained_model, T_val_loader, 7)
#     accuracy_t_list.append(accuracy_t)
#     pr_t_list.append(pr_t)
#     re_t_list.append(re_t)
#     f1_t_list.append(f1_t)

# # Calculate mean and standard deviation of performance metrics
# mean_accuracy_s = np.mean(accuracy_s_list)
# mean_pr_s = np.mean(pr_s_list)
# mean_re_s = np.mean(re_s_list)
# mean_f1_s = np.mean(f1_s_list)

# mean_accuracy_t = np.mean(accuracy_t_list)
# mean_pr_t = np.mean(pr_t_list)
# mean_re_t = np.mean(re_t_list)
# mean_f1_t = np.mean(f1_t_list)

# print(f"Mean Accuracy on Source Domain: {mean_accuracy_s*100:.2f}, Precision: {mean_pr_s*100:.2f}, Recall: {mean_re_s*100:.2f}, F1 Score: {mean_f1_s*100:.2f}")
# print(f"Mean Accuracy on Target Domain: {mean_accuracy_t*100:.2f}, Precision: {mean_pr_t*100:.2f}, Recall: {mean_re_t*100:.2f}, F1 Score: {mean_f1_t*100:.2f}")







#%% ORIGINAL 

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in S_train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#     epoch_loss = running_loss / len(S_train_loader.dataset)
#     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

# # Evaluate the model
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in S_val_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# val_accuracy = correct / total
# print(f'Validation Accuracy: {val_accuracy:.4f}')


# # %%

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import precision_score,f1_score,recall_score
# import seaborn as sns

# def evaluate_and_plot_confusion_matrix(model, loader, title, num_classes):
#     model.eval()
#     true_labels = []
#     predictions = []
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             class_outputs = model(inputs)  # Alpha set to 0 during inference
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
#     plt.figure(figsize=(8,6),dpi=300)
#     sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes_of_interest_names,
#                 yticklabels=classes_of_interest_names)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()

#     return accuracy,precision,recall,f1

# # Call the evaluation function for both validation sets
# accuracy_s,pr_s,re_s,f1_s = evaluate_and_plot_confusion_matrix(model, S_val_loader, "Source Domain", 7)
# accuracy_t,pr_t,re_t,f1_t = evaluate_and_plot_confusion_matrix(model, T_val_loader, "Target Domain", 7)
# print(f"Accuracy on Source Domain: {accuracy_s*100:.2f} pr {pr_s*100:.2f} re {re_s*100:.2f} f1 {f1_s*100:.2f}")
# print(f"Accuracy on Target Domain: {accuracy_t*100:.2f} pr {pr_t*100:.2f} re {re_t*100:.2f} f1 {f1_t*100:.2f}")