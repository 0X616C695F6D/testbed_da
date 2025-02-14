"""
Models for baseline comparison.
No DA method used.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class CNN(nn.Module):
    def __init__(self, output_dim=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 256, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 256)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2048, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
  
    
# !bidirectional
class CLDNN(nn.Module):
    def __init__(self, output_dim=7):
        super(CLDNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(508 * 64, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x = x.contiguous().view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Base:
    def __init__(self, model_cls, device, S_train_loader, S_val_loader, T_val_loader,
                 class_subset, n_classes, lr=0.001, n_epochs=50, n_runs=10, patience=5, step_size=10, gamma=0.1):
        # model & parameters
        self.model_cls = model_cls
        self.device = device
        self.class_subset = class_subset
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_runs = n_runs
        self.patience = patience
        self.step_size = step_size
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss()

        # dataloaders
        self.S_train_loader = S_train_loader
        self.S_val_loader = S_val_loader
        self.T_val_loader = T_val_loader

        # source metrics
        self.accuracy_s_list = []
        self.pr_s_list = []
        self.re_s_list = []
        self.f1_s_list = []

        # target metrics
        self.accuracy_t_list = []
        self.pr_t_list = []
        self.re_t_list = []
        self.f1_t_list = []

        # accuracies
        self.class_accuracies_s = np.zeros((n_runs, n_classes))
        self.class_accuracies_t = np.zeros((n_runs, n_classes))
    
    def train_model(self, model, optimizer, scheduler):
        best_val_loss = float('inf')
        trigger_times = 0
        for epoch in range(self.n_epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.S_train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(self.S_train_loader.dataset)
            train_accuracy = correct / total
    
            # Validation
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.S_val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss = val_running_loss / len(self.S_val_loader.dataset)
            val_accuracy = correct / total
    
            print(f'Epoch {epoch+1}/{self.n_epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
            scheduler.step()
    
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print('Early stopping!')
                    break
    
        return model
        
    def eva_model(self, model, loader, n_classes):
        # Evaluate model
        model.eval()
        true_labels = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
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
    
        #Plot the confusion matrix
        #plt.figure(figsize=(8,6),dpi=300)
        #sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
        #            xticklabels=n_classes,
        #            yticklabels=n_classes)
        #plt.yticks(fontsize=14,rotation=360)
        #plt.xticks(fontsize=14,rotation=90)
        #plt.title('Confusion Matrix')
        #plt.show()
    
        return accuracy, precision, recall, f1, class_accuracy

    def run(self):
        for run in range(self.n_runs):
            print(f'\nRun {run+1}/{self.n_runs}')
            # Model is reset per run. Change model type here too.
            model = self.model_cls().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
            trained_model = self.train_model(model, optimizer, scheduler)
        
            # Evaluate on source domain
            accuracy_s, pr_s, re_s, f1_s, class_acc_s = self.eva_model(trained_model, self.S_val_loader, self.n_classes)
            self.accuracy_s_list.append(accuracy_s)
            self.pr_s_list.append(pr_s)
            self.re_s_list.append(re_s)
            self.f1_s_list.append(f1_s)
            self.class_accuracies_s[run] = class_acc_s
            
            # Evaluate on target domain
            accuracy_t, pr_t, re_t, f1_t, class_acc_t = self.eva_model(trained_model, self.T_val_loader, self.n_classes)
            self.accuracy_t_list.append(accuracy_t)
            self.pr_t_list.append(pr_t)
            self.re_t_list.append(re_t)
            self.f1_t_list.append(f1_t)
            self.class_accuracies_t[run] = class_acc_t
        
        # Calculate mean and standard deviation of performance metrics
        mean_accuracy_s = np.mean(self.accuracy_s_list)
        mean_pr_s = np.mean(self.pr_s_list)
        mean_re_s = np.mean(self.re_s_list)
        mean_f1_s = np.mean(self.f1_s_list)
        
        mean_accuracy_t = np.mean(self.accuracy_t_list)
        mean_pr_t = np.mean(self.pr_t_list)
        mean_re_t = np.mean(self.re_t_list)
        mean_f1_t = np.mean(self.f1_t_list)
        
        mean_class_accuracies_s = np.mean(self.class_accuracies_s, axis=0)
        mean_class_accuracies_t = np.mean(self.class_accuracies_t, axis=0)
        
        print(f"\nSource performance: {mean_accuracy_s*100:.2f} {mean_pr_s*100:.2f} {mean_re_s*100:.2f} {mean_f1_s*100:.2f}")
        print(f"Target performance: {mean_accuracy_t*100:.2f} {mean_pr_t*100:.2f} {mean_re_t*100:.2f} {mean_f1_t*100:.2f}\n")
        
        for i, class_name in enumerate(self.class_subset):
            print(f"{class_name}: {mean_class_accuracies_t[i]*100:.2f}")

        return mean_accuracy_s, mean_accuracy_t