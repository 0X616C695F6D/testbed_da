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
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_bottleneck = nn.Linear(32512, 512) # bottleneck layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_bottleneck(x) # bottleneck layer
        return x

class CLDNN_C_JAN(nn.Module):
    def __init__(self, output_dim):
        super(CLDNN_C_JAN, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x, return_intermediate=False):
        inter = torch.relu(self.fc1(x))
        out = self.fc2(inter)
        if return_intermediate:
            return out, inter
        else:
            return out

class Jan:
    def __init__(self, num_classes, device, 
                 S_train_loader, T_train_loader, 
                 S_val_loader, T_val_loader,
                 n_epochs=50, lr=0.001, lambda_jmmd=0.1, n_runs=10,
                 early_stopping_patience=5):
        
        self.num_classes = num_classes
        self.device = device
        self.S_train_loader = S_train_loader
        self.T_train_loader = T_train_loader
        self.S_val_loader = S_val_loader
        self.T_val_loader = T_val_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_jmmd = lambda_jmmd
        self.n_runs = n_runs
        self.early_stopping_patience = early_stopping_patience

    @staticmethod
    def mmd_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (total.size(0)**2 - total.size(0))
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def jmmd_loss(source_list, target_list, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # Concatenate each list's elements along the feature dimension.
        source_joint = torch.cat([s.view(s.size(0), -1) for s in source_list], dim=1)
        target_joint = torch.cat([t.view(t.size(0), -1) for t in target_list], dim=1)
        
        kernels = Jan.mmd_kernel(source_joint, target_joint, kernel_mul, kernel_num, fix_sigma)
        ns = source_joint.size(0)
        nt = target_joint.size(0)
        XX = kernels[:ns, :ns]
        YY = kernels[ns:, ns:]
        XY = kernels[:ns, ns:]
        YX = kernels[ns:, :ns]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    
    def train_model(self):
        feature_extractor = CLDNN_G().to(self.device)
        classifier = CLDNN_C_JAN(output_dim=self.num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        best_val_loss = float('inf')
        trigger_times = 0
        
        for epoch in range(self.n_epochs):
            feature_extractor.train()
            classifier.train()
            
            running_classification_loss = 0.0
            running_jmmd_loss = 0.0
            
            source_iter = iter(self.S_train_loader)
            target_iter = iter(self.T_train_loader)
            num_batches = min(len(self.S_train_loader), len(self.T_train_loader))
            
            for batch_idx in range(num_batches):
                # Get source batch
                try:
                    inputs_s, labels_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.S_train_loader)
                    inputs_s, labels_s = next(source_iter)
                    
                # Get target batch
                try:
                    inputs_t, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.T_train_loader)
                    inputs_t, _ = next(target_iter)
                
                inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                inputs_t = inputs_t.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass for source and target
                features_s = feature_extractor(inputs_s)
                features_t = feature_extractor(inputs_t)
                outputs_s, inter_s = classifier(features_s, return_intermediate=True)
                outputs_t, inter_t = classifier(features_t, return_intermediate=True)
                
                # Compute losses
                classification_loss = criterion(outputs_s, labels_s)
                loss_jmmd = self.jmmd_loss([features_s, inter_s], [features_t, inter_t])
                
                total_loss = classification_loss + self.lambda_jmmd * loss_jmmd
                total_loss.backward()
                optimizer.step()
                
                running_classification_loss += classification_loss.item()
                running_jmmd_loss += loss_jmmd.item()
            
            scheduler.step()
            avg_classification_loss = running_classification_loss / num_batches
            avg_jmmd_loss = running_jmmd_loss / num_batches
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Class Loss: {avg_classification_loss:.4f}, JMMD Loss: {avg_jmmd_loss:.4f}')
            
            # Validation on source domain
            feature_extractor.eval()
            classifier.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs_s, labels_s in self.S_val_loader:
                    inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                    features_s = feature_extractor(inputs_s)
                    outputs_s, _ = classifier(features_s, return_intermediate=True)
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
                if trigger_times >= self.early_stopping_patience:
                    print('Early stopping!')
                    break
            
        return feature_extractor, classifier
    
    def evaluate_model(self, feature_extractor, classifier, loader):
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
        per_class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
        return accuracy, precision, recall, f1, per_class_accuracy
    
    def run(self):
        accuracy_s_list, pr_s_list, re_s_list, f1_s_list = [], [], [], []
        accuracy_t_list, pr_t_list, re_t_list, f1_t_list = [], [], [], []
        class_accuracies_s = []
        class_accuracies_t = []
        
        for run in range(self.n_runs):
            print(f'\nRun {run+1}/{self.n_runs}')
            feature_extractor, classifier = self.train_model()
            
            # Evaluate on source domain
            accuracy_s, pr_s, re_s, f1_s, class_acc_s = self.evaluate_model(feature_extractor, classifier, self.S_val_loader)
            print(f'Source Domain Performance - Accuracy: {accuracy_s*100:.2f}%, '
                  f'Precision: {pr_s*100:.2f}%, Recall: {re_s*100:.2f}%, F1 Score: {f1_s*100:.2f}%')
            accuracy_s_list.append(accuracy_s)
            pr_s_list.append(pr_s)
            re_s_list.append(re_s)
            f1_s_list.append(f1_s)
            class_accuracies_s.append(class_acc_s)
            
            # Evaluate on target domain
            accuracy_t, pr_t, re_t, f1_t, class_acc_t = self.evaluate_model(feature_extractor, classifier, self.T_val_loader)
            print(f'Target Domain Performance - Accuracy: {accuracy_t*100:.2f}%, '
                  f'Precision: {pr_t*100:.2f}%, Recall: {re_t*100:.2f}%, F1 Score: {f1_t*100:.2f}%')
            accuracy_t_list.append(accuracy_t)
            pr_t_list.append(pr_t)
            re_t_list.append(re_t)
            f1_t_list.append(f1_t)
            class_accuracies_t.append(class_acc_t)
        
        # Calculate mean performance metrics across runs
        mean_accuracy_s = np.mean(accuracy_s_list)
        mean_pr_s     = np.mean(pr_s_list)
        mean_re_s     = np.mean(re_s_list)
        mean_f1_s     = np.mean(f1_s_list)
        
        mean_accuracy_t = np.mean(accuracy_t_list)
        mean_pr_t     = np.mean(pr_t_list)
        mean_re_t     = np.mean(re_t_list)
        mean_f1_t     = np.mean(f1_t_list)
                
        mean_class_accuracies_s = np.mean(class_accuracies_s, axis=0)
        mean_class_accuracies_t = np.mean(class_accuracies_t, axis=0)
        
        print(f"\nSource performance: {mean_accuracy_s*100:.2f}% {mean_pr_s*100:.2f}% {mean_re_s*100:.2f}% {mean_f1_s*100:.2f}%")
        print(f"Target performance: {mean_accuracy_t*100:.2f}% {mean_pr_t*100:.2f}% {mean_re_t*100:.2f}% {mean_f1_t*100:.2f}%")
        
        print("\nPer-Class Accuracy on Target Domain (Mean over runs):")
        for i, acc in enumerate(mean_class_accuracies_t):
            print(f"  Class {i}: {acc*100:.2f}%")

        return mean_accuracy_s, mean_accuracy_t
