"""
Implements CORAL (Correlation alignment for UDA)

Contains train, test and run functions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


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

        
class Coral:
    def __init__(self, G, C, device, S_train_loader, S_val_loader, T_train_loader, T_val_loader,
                 class_subset, n_classes, lr=0.001, n_epochs=50, n_runs=10, patience=5, lambda_coral=1):
        # model & parameters
        self.G = G
        self.C = C
        self.device = device
        self.class_subset = class_subset
        self.num_classes = n_classes
        self.learning_rate = lr
        self.n_epochs = n_epochs
        self.n_runs = n_runs
        self.patience = patience
        self.lambda_coral = lambda_coral

        # dataloaders
        self.S_train_loader = S_train_loader
        self.S_val_loader = S_val_loader
        self.T_train_loader = T_train_loader
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
        self.class_accuracies_s = []
        self.class_accuracies_t = []

    def compute_covariance(self, input_data):
        n = input_data.size(0)
        mean = torch.mean(input_data, dim=0, keepdim=True)
        input_data_centered = input_data - mean
        cov = (input_data_centered.t() @ input_data_centered) / (n - 1)
        return cov


    def coral_comp(self, source, target):
        d = source.size(1)
        
        source_cov = self.compute_covariance(source)
        target_cov = self.compute_covariance(target)
        
        loss = torch.sum((source_cov - target_cov) ** 2)
        loss = loss / (4 * d * d)
        return loss
    
    def train_model(self):
        feature_extractor = self.G().to(self.device)
        classifier = self.C(output_dim=self.num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        best_val_loss = float('inf')
        trigger_times = 0
        
        for epoch in range(self.n_epochs):
            feature_extractor.train()
            classifier.train()
            
            running_classification_loss = 0.0
            running_coral_loss = 0.0
            
            source_iter = iter(self.S_train_loader)
            target_iter = iter(self.T_train_loader)
            num_batches = min(len(self.S_train_loader), len(self.T_train_loader))
            
            for batch_idx in range(num_batches):
                try:
                    inputs_s, labels_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.S_train_loader)
                    inputs_s, labels_s = next(source_iter)
                try:
                    inputs_t, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.T_train_loader)
                    inputs_t, _ = next(target_iter)
                
                inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                inputs_t = inputs_t.to(self.device)
                
                optimizer.zero_grad()
                
                features_s = feature_extractor(inputs_s)
                features_t = feature_extractor(inputs_t)
                outputs_s = classifier(features_s)
                
                classification_loss = criterion(outputs_s, labels_s)
                
                coral_loss_value = self.coral_comp(features_s, features_t)
                
                total_loss = classification_loss + self.lambda_coral * coral_loss_value
                
                total_loss.backward()
                optimizer.step()
                
                running_classification_loss += classification_loss.item()
                running_coral_loss += coral_loss_value.item()
            
            scheduler.step()
            
            avg_classification_loss = running_classification_loss / num_batches
            avg_coral_loss = running_coral_loss / num_batches
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Class Loss: {avg_classification_loss:.4f}, CORAL Loss: {avg_coral_loss:.4f}')
            
            feature_extractor.eval()
            classifier.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs_s, labels_s in self.S_val_loader:
                    inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                    features_s = feature_extractor(inputs_s)
                    outputs_s = classifier(features_s)
                    loss_s = criterion(outputs_s, labels_s)
                    val_loss += loss_s.item() * inputs_s.size(0)
                    total_samples += inputs_s.size(0)
            val_loss = val_loss / total_samples
            print(f'Validation Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print('Early stopping!')
                    break
                
        return feature_extractor, classifier

    def evaluate_model(self, feature_extractor, classifier, loader, num_classes):
        feature_extractor.eval()
        classifier.eval()
        true_labels = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                features = feature_extractor(inputs)
                outputs = classifier(features)
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        conf_mat = confusion_matrix(true_labels, predictions)
        
        #plt.figure(figsize=(8,6), dpi=300)
        #sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
         #           xticklabels=class_subset,
         #           yticklabels=class_subset)
        #plt.yticks(fontsize=14, rotation=360)
        #plt.xticks(fontsize=14, rotation=90)
        #plt.title('Confusion Matrix')
        #plt.show()
        
        class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
        return accuracy, precision, recall, f1, class_accuracy

    def run(self):
        for run in range(self.n_runs):
            print(f'\nRun {run+1}/{self.n_runs}')
            feature_extractor, classifier = self.train_model()
            
            # Evaluate on source domain
            accuracy_s, pr_s, re_s, f1_s, class_acc_s = self.evaluate_model(feature_extractor, classifier, self.S_val_loader, self.num_classes)
            print(f'Source Domain Performance - Accuracy: {accuracy_s*100:.2f}%, Precision: {pr_s*100:.2f}%, Recall: {re_s*100:.2f}%, F1 Score: {f1_s*100:.2f}%')
            self.accuracy_s_list.append(accuracy_s)
            self.pr_s_list.append(pr_s)
            self.re_s_list.append(re_s)
            self.f1_s_list.append(f1_s)
            self.class_accuracies_s.append(class_acc_s)
            
            # Evaluate on target domain
            accuracy_t, pr_t, re_t, f1_t, class_acc_t = self.evaluate_model(feature_extractor, classifier, self.T_val_loader, self.num_classes)
            print(f'Target Domain Performance - Accuracy: {accuracy_t*100:.2f}%, Precision: {pr_t*100:.2f}%, Recall: {re_t*100:.2f}%, F1 Score: {f1_t*100:.2f}%')
            self.accuracy_t_list.append(accuracy_t)
            self.pr_t_list.append(pr_t)
            self.re_t_list.append(re_t)
            self.f1_t_list.append(f1_t)
            self.class_accuracies_t.append(class_acc_t)
        
        # Calculate mean performance metrics across runs
        mean_accuracy_s = np.mean(self.accuracy_s_list)
        mean_pr_s     = np.mean(self.pr_s_list)
        mean_re_s     = np.mean(self.re_s_list)
        mean_f1_s     = np.mean(self.f1_s_list)
        
        mean_accuracy_t = np.mean(self.accuracy_t_list)
        mean_pr_t     = np.mean(self.pr_t_list)
        mean_re_t     = np.mean(self.re_t_list)
        mean_f1_t     = np.mean(self.f1_t_list)
        
        mean_class_accuracies_s = np.mean(np.array(self.class_accuracies_s), axis=0)
        mean_class_accuracies_t = np.mean(np.array(self.class_accuracies_t), axis=0)
        
        
        print(f"\nSource performance: {mean_accuracy_s*100:.2f}% {mean_pr_s*100:.2f}% {mean_re_s*100:.2f}% {mean_f1_s*100:.2f}%")
        print(f"Target performance: {mean_accuracy_t*100:.2f}% {mean_pr_t*100:.2f}% {mean_re_t*100:.2f}% {mean_f1_t*100:.2f}%")
        
        print("\nPer-Class Accuracy on Target Domain:")
        for i, class_name in enumerate(self.class_subset):  # Ensure class_subset is defined
            print(f"{class_name}: {mean_class_accuracies_t[i]*100:.2f}%")

        return mean_accuracy_s, mean_accuracy_t