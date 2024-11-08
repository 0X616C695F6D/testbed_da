import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import funcs
import dann
import base
import plots
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score
from torch.optim.lr_scheduler import StepLR

#%% Prelim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'

# Set of classes in RadioML 2018.01A
classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
           'FM', 'GMSK', 'OQPSK']

class_subset = ['16QAM', '64QAM', '8ASK', 'BPSK', 'QPSK', '8PSK','GMSK']
# class_subset = ['256QAM', '64QAM', 'OQPSK', 'BPSK', 'QPSK', 'FM','GMSK']
# class_subset = ['8ASK', 'BPSK', '32QAM', '16QAM', 'AM-DSB-SC','AM-SSB-SC']
# class_subset = ['8ASK', 'BPSK','8PSK', '16PSK','32QAM', '64QAM','OQPSK']
# class_subset = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
#             '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
#             '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
#             'FM', 'GMSK', 'OQPSK']
n_classes = len(class_subset)

# Source dataloader
indices = [classes.index(name) for name in class_subset]
X_path, Y_path = funcs.filter_and_save_data(file_path, classes, class_subset,
                                            z_value=20, domain='S')
S_train_loader, S_val_loader = funcs.create_loader(X_path, Y_path)

# Target dataloader
X_path, Y_path = funcs.filter_and_save_data(file_path, classes, class_subset,
                                            z_value=10, domain='T')
T_train_loader, T_val_loader = funcs.create_loader(X_path, Y_path)

# Plot of a waveform
# plots.plot_signal(X_path, signal_index=12221)


#%% Baseline

# Hyperparameters
lr = 0.001
n_epochs = 25
n_runs = 10

model = base.MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=.1)

# Initialize lists to store performance metrics for each run
accuracy_s_list, pr_s_list, re_s_list, f1_s_list = [], [], [], []
accuracy_t_list, pr_t_list, re_t_list, f1_t_list = [], [], [], []
class_accuracies_s = np.zeros((n_runs, len(class_subset)))
class_accuracies_t = np.zeros((n_runs, len(class_subset)))

def train_model():
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0
    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in S_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(S_train_loader.dataset)
        train_accuracy = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in S_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_running_loss / len(S_val_loader.dataset)
        val_accuracy = correct / total

        print(f'Epoch {epoch+1}/{n_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    return model

def eva_model(model, loader, num_classes):
    # Evaluate model
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_outputs = model(inputs)
            _, preds = torch.max(class_outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # Performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    
    # Confusion matrix and per-class accuracy
    conf_mat = confusion_matrix(true_labels, predictions)
    class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)

    # Plot the confusion matrix
    plt.figure(figsize=(8,6),dpi=300)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_subset,
                yticklabels=class_subset)
    plt.yticks(fontsize=14,rotation=360)
    plt.xticks(fontsize=14,rotation=90)
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, f1, class_accuracy

for run in range(n_runs):
    print(f'\nRun {run+1}/{n_runs}')
    model = base.CNN().to(device) # Reset model and optimizer for each run
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trained_model = train_model()

    # Evaluate on source domain
    accuracy_s, pr_s, re_s, f1_s, class_acc_s = eva_model(trained_model, S_val_loader, n_classes)
    accuracy_s_list.append(accuracy_s)
    pr_s_list.append(pr_s)
    re_s_list.append(re_s)
    f1_s_list.append(f1_s)
    class_accuracies_s[run] = class_acc_s
    
    # Evaluate on target domain
    accuracy_t, pr_t, re_t, f1_t, class_acc_t = eva_model(trained_model, T_val_loader, n_classes)
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

print(f"\nSource performance: {mean_accuracy_s*100:.2f} {mean_pr_s*100:.2f} {mean_re_s*100:.2f} {mean_f1_s*100:.2f}")
print(f"Target performance: {mean_accuracy_t*100:.2f} {mean_pr_t*100:.2f} {mean_re_t*100:.2f} {mean_f1_t*100:.2f}\n")

for i, class_name in enumerate(class_subset):
    print(f"{class_name}: {mean_class_accuracies_t[i]*100:.2f}")


#%% DANN



#%% MCD


