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

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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

class DAN:
    def __init__(self, DANN_cls, FA, LP, DC, device, S_train_loader, S_val_loader, T_train_loader, T_val_loader,
                 class_subset, n_classes, lr=0.001, n_epochs=50, n_runs=5):
        # model & parameters
        self.DANN_cls = DANN_cls
        self.FA = FA
        self.LP = LP
        self.DC = DC
        self.device = device
        self.class_subset = class_subset
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_runs = n_runs

        # dataloaders
        self.S_train_loader = S_train_loader
        self.S_val_loader = S_val_loader
        self.T_val_loader = T_val_loader
        self.T_train_loader = T_val_loader

        # metrics are included in the run function

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_domain = nn.CrossEntropyLoss()

    def train_dann(self, model, optimizer):
        for epoch in range(self.n_epochs):
            model.train()
            total_loss, total_domain_loss, total_class_loss = 0, 0, 0
            len_dataloader = min(len(self.S_train_loader), len(self.T_train_loader))
            data_source_iter = iter(self.S_train_loader)
            data_target_iter = iter(self.T_train_loader)
    
            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / self.n_epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
    
                # Training model using source data
                s_data, s_label = next(data_source_iter)
                s_data, s_label = s_data.to(self.device), s_label.to(self.device)
                optimizer.zero_grad()
                class_output, domain_output = model(s_data, alpha)
                err_s_label = self.criterion_class(class_output, s_label)
                err_s_domain = self.criterion_domain(domain_output, torch.zeros(s_data.size(0), dtype=torch.long).to(self.device))
    
                # Training model using target data
                t_data, _ = next(data_target_iter)
                t_data = t_data.to(self.device)
                _, domain_output = model(t_data, alpha)
                err_t_domain = self.criterion_domain(domain_output, torch.ones(t_data.size(0), dtype=torch.long).to(self.device))
    
                # Combining the losses
                loss = err_s_label + err_s_domain + err_t_domain
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                total_domain_loss += err_s_domain.item() + err_t_domain.item()
                total_class_loss += err_s_label.item()
    
            print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len_dataloader:.4f}, Domain Loss: {total_domain_loss/len_dataloader:.4f}, Class Loss: {total_class_loss/len_dataloader:.4f}')

    def evaluate_and_plot_confusion_matrix(self, model, loader, title, num_classes):
        model.eval()
        true_labels = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                class_outputs, _ = model(inputs, alpha=0)
                _, preds = torch.max(class_outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
    
        # Calculate overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        
        # Confusion matrix & per-class accuracy
        conf_mat = confusion_matrix(true_labels, predictions)
        per_class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
        
        # Plot the confusion matrix
        #plt.figure(figsize=(8,6), dpi=300)
        #sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
        #            xticklabels=class_subset,
        #            yticklabels=class_subset)
        #plt.yticks(fontsize=14,rotation=360)
        #plt.xticks(fontsize=14,rotation=90)
        #plt.title(f'Confusion Matrix - {title}')
        #plt.show()
        
        return accuracy, precision, recall, f1, per_class_accuracy


    def run(self):
        source_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'per_class_accuracy': []}
        target_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'per_class_accuracy': []}
        
        for _ in range(self.n_runs):
            # Model is reset per run. Change model type here too.
            model = self.DANN_cls(self.FA,self.LP,self.DC).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
            self.train_dann(model, optimizer)
        
            # Evaluate on source domain
            accuracy_s, pr_s, re_s, f1_s, per_class_accuracy_s = self.evaluate_and_plot_confusion_matrix(model, self.S_val_loader, "Source Domain", self.n_classes)
            source_metrics['accuracy'].append(accuracy_s)
            source_metrics['precision'].append(pr_s)
            source_metrics['recall'].append(re_s)
            source_metrics['f1'].append(f1_s)
            source_metrics['per_class_accuracy'].append(per_class_accuracy_s)
        
            # Evaluate on target domain
            accuracy_t, pr_t, re_t, f1_t, per_class_accuracy_t = self.evaluate_and_plot_confusion_matrix(model, self.T_val_loader, "Target Domain", self.n_classes)
            print(f'{accuracy_t*100:.2f}\n\n')
            target_metrics['accuracy'].append(accuracy_t)
            target_metrics['precision'].append(pr_t)
            target_metrics['recall'].append(re_t)
            target_metrics['f1'].append(f1_t)
            target_metrics['per_class_accuracy'].append(per_class_accuracy_t)
        
        # Calculate and print average metrics
        avg_source_metrics = {metric: np.mean(values) for metric, values in source_metrics.items() if metric != 'per_class_accuracy'}
        avg_target_metrics = {metric: np.mean(values) for metric, values in target_metrics.items() if metric != 'per_class_accuracy'}
        
        print("Source performance:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            value = avg_source_metrics.get(metric, 0)
            print(f"{value*100:.2f}", end= ' ')
            
        print("\nTarget performance:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            value = avg_target_metrics.get(metric, 0)
            print(f"{value*100:.2f}", end= ' ')
        
        avg_target_per_class_accuracy = np.mean(np.array(target_metrics['per_class_accuracy']), axis=0)
        print("\n\nPer-class target performance:", end=' ')
        for acc in avg_target_per_class_accuracy:
            print(f"{acc*100:.2f}", end=' ')

        return accuracy_s, accuracy_t