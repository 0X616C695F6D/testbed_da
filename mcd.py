"""
File that implements MCD (Maximum Classifier Discrepancy).

Contains train, test, run functions.
Only model available is CLDNN.
Also has GRL for MCD.
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


#%% GRL for MCD
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class CLDNN_C(nn.Module):
    def __init__(self, output_dim):
        super(CLDNN_C, self).__init__()
        self.fc1 = nn.Linear(508 * 64, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x, reverse=False, lambda_=1.0):
        if reverse:
            x = grad_reverse(x, lambda_)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Mcd:   
    def __init__(self, G, C, device, S_train_loader, S_val_loader, T_train_loader, T_val_loader,
                 class_subset, n_classes, lr=0.001, n_epochs=50, n_runs=10, patience=5):
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

    def discrepancy_loss(self, output1, output2):
        return torch.mean(torch.abs(F.softmax(output1, dim=1) - F.softmax(output2, dim=1)))

    def evaluate_model(self, feature_extractor, classifier1, classifier2, loader, num_classes):
        feature_extractor.eval()
        classifier1.eval()
        classifier2.eval()
        true_labels = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                features = feature_extractor(inputs)
                outputs1 = classifier1(features)
                outputs2 = classifier2(features)
                outputs = (outputs1 + outputs2) / 2
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        conf_mat = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        #plt.figure(figsize=(8,6), dpi=300)
        #sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
        #            xticklabels=class_subset,
        #            yticklabels=class_subset)
        #plt.yticks(fontsize=14, rotation=360)
        #plt.xticks(fontsize=14, rotation=90)
        #plt.title('Confusion Matrix')
        #plt.show()
        
        class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
        return accuracy, precision, recall, f1, class_accuracy

    def train_model(self):
        # Initialize models
        feature_extractor = self.G().to(self.device)
        classifier1 = self.C(output_dim=self.num_classes).to(self.device)
        classifier2 = self.C(output_dim=self.num_classes).to(self.device)
        
        # Define criterion and optimizers
        criterion = nn.CrossEntropyLoss()
        optimizer_g = optim.Adam(feature_extractor.parameters(), lr=self.learning_rate)
        optimizer_c1 = optim.Adam(classifier1.parameters(), lr=self.learning_rate)
        optimizer_c2 = optim.Adam(classifier2.parameters(), lr=self.learning_rate)
        
        # Learning rate schedulers
        scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
        scheduler_c1 = optim.lr_scheduler.StepLR(optimizer_c1, step_size=10, gamma=0.1)
        scheduler_c2 = optim.lr_scheduler.StepLR(optimizer_c2, step_size=10, gamma=0.1)
        
        # Early stopping parameters
        best_val_loss = float('inf')
        trigger_times = 0
        
        for epoch in range(self.n_epochs):
            feature_extractor.train()
            classifier1.train()
            classifier2.train()
            
            running_loss_s = 0.0
            running_loss_dis = 0.0
            
            source_iter = iter(self.S_train_loader)
            target_iter = iter(self.T_train_loader)
            num_batches = min(len(self.S_train_loader), len(self.T_train_loader))
            
            for batch_idx in range(num_batches):
                # Get source batch
                inputs_s, labels_s = next(source_iter)
                inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                
                # Get target batch
                inputs_t, _ = next(target_iter)
                inputs_t = inputs_t.to(self.device)
                
                # Combine source and target data
                inputs = torch.cat([inputs_s, inputs_t], dim=0)
                
                # Zero the parameter gradients
                optimizer_g.zero_grad()
                optimizer_c1.zero_grad()
                optimizer_c2.zero_grad()
                
                # Forward pass
                features = feature_extractor(inputs)
                features_s = features[:inputs_s.size(0)]
                features_t = features[inputs_s.size(0):]
                
                # Classification outputs for source data
                outputs_s1 = classifier1(features_s)
                outputs_s2 = classifier2(features_s)
                
                # Outputs for target data with gradient reversal
                outputs_t1 = classifier1(features_t, reverse=True, lambda_=1.0)
                outputs_t2 = classifier2(features_t, reverse=True, lambda_=1.0)
                
                # Compute losses
                loss_s1 = criterion(outputs_s1, labels_s)
                loss_s2 = criterion(outputs_s2, labels_s)
                loss_s = loss_s1 + loss_s2
                
                loss_dis = self.discrepancy_loss(outputs_t1, outputs_t2)
                
                total_loss = loss_s + loss_dis
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier2.parameters(), max_norm=1.0)
                
                # Optimization step
                optimizer_g.step()
                optimizer_c1.step()
                optimizer_c2.step()
                
                running_loss_s += loss_s.item()
                running_loss_dis += loss_dis.item()
            
            # Learning rate scheduler step
            scheduler_g.step()
            scheduler_c1.step()
            scheduler_c2.step()
            
            # Print average losses for the epoch
            avg_loss_s = running_loss_s / num_batches
            avg_loss_dis = running_loss_dis / num_batches
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Class Loss: {avg_loss_s:.4f}, Discrepancy Loss: {avg_loss_dis:.4f}')
            
            # Early stopping based on validation loss on source domain
            feature_extractor.eval()
            classifier1.eval()
            classifier2.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs_s, labels_s in self.S_val_loader:
                    inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                    features_s = feature_extractor(inputs_s)
                    outputs_s1 = classifier1(features_s)
                    outputs_s2 = classifier2(features_s)
                    loss_s1 = criterion(outputs_s1, labels_s)
                    loss_s2 = criterion(outputs_s2, labels_s)
                    loss_s = loss_s1 + loss_s2
                    val_loss += loss_s.item() * inputs_s.size(0)
                    total_samples += inputs_s.size(0)
            val_loss = val_loss / total_samples
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print('Early stopping!')
                    break
    
        return feature_extractor, classifier1, classifier2

    # Run multiple times and collect performance metrics
    def run(self):
        for run in range(self.n_runs):
            print(f'\nRun {run+1}/{self.n_runs}')
            feature_extractor, classifier1, classifier2 = self.train_model()
            
            # Evaluate on source domain
            accuracy_s, pr_s, re_s, f1_s, class_acc_s = self.evaluate_model(feature_extractor, classifier1, classifier2, self.S_val_loader, self.num_classes)
            print(f'Source Domain Performance - Accuracy: {accuracy_s*100:.2f}%, Precision: {pr_s*100:.2f}%, Recall: {re_s*100:.2f}%, F1 Score: {f1_s*100:.2f}%')
            self.accuracy_s_list.append(accuracy_s)
            self.pr_s_list.append(pr_s)
            self.re_s_list.append(re_s)
            self.f1_s_list.append(f1_s)
            self.class_accuracies_s.append(class_acc_s)
            
            # Evaluate on target domain
            accuracy_t, pr_t, re_t, f1_t, class_acc_t = self.evaluate_model(feature_extractor, classifier1, classifier2, self.T_val_loader, self.num_classes)
            print(f'Target Domain Performance - Accuracy: {accuracy_t*100:.2f}%, Precision: {pr_t*100:.2f}%, Recall: {re_t*100:.2f}%, F1 Score: {f1_t*100:.2f}%')
            self.accuracy_t_list.append(accuracy_t)
            self.pr_t_list.append(pr_t)
            self.re_t_list.append(re_t)
            self.f1_t_list.append(f1_t)
            self.class_accuracies_t.append(class_acc_t)
        
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
        
        print(f"\nSource performance: {mean_accuracy_s*100:.2f}% {mean_pr_s*100:.2f}% {mean_re_s*100:.2f}% {mean_f1_s*100:.2f}%")
        print(f"Target performance: {mean_accuracy_t*100:.2f}% {mean_pr_t*100:.2f}% {mean_re_t*100:.2f}% {mean_f1_t*100:.2f}%")
        
        print("\nPer-Class Accuracy on Target Domain:")
        for i, class_name in enumerate(self.class_subset):
            print(f"{class_name}: {mean_class_accuracies_t[i]*100:.2f}%")

        return mean_accuracy_s, mean_accuracy_t