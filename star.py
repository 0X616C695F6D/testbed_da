"""
File that implement ML models & train & test & run functions for STAR (stochastic classifier).
https://github.com/zhiheLu/STAR_Stochastic_Classifiers_for_UDA.
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

#%% CNN

class CNN_G(nn.Module):
    def __init__(self):
        super(CNN_G, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 256)
        x = F.relu(self.fc1(x))
        return x

class CNN_C(nn.Module):
    def __init__(
        self,
        output_dim,
        num_classifiers_train=2,
        num_classifiers_test=20,
        init='kaiming_u',
        use_init=False
    ):
        super(CNN_C, self).__init__()
        self.num_classifiers_train = num_classifiers_train
        self.num_classifiers_test = num_classifiers_test
        self.init = init

        function_init = {
            'kaiming_u': nn.init.kaiming_uniform_,
            'kaiming_n': nn.init.kaiming_normal_,
            'xavier': nn.init.xavier_normal_
        }

        self.fc1 = nn.Linear(128, 64)
        self.bn1_fc = nn.BatchNorm1d(64)

        self.mu2 = nn.Parameter(torch.randn(output_dim, 64))
        self.sigma2 = nn.Parameter(torch.zeros(output_dim, 64))

        if use_init:
            all_parameters = [self.mu2, self.sigma2]
            for item in all_parameters:
                function_init[self.init](item)

        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, only_mu=True):
        x = F.relu(self.bn1_fc(self.fc1(x)))

        sigma2_pos = torch.sigmoid(self.sigma2)
        fc2_distribution = torch.distributions.Normal(self.mu2, sigma2_pos)

        if self.training:
            classifiers = []
            for _ in range(self.num_classifiers_train):
                fc2_w = fc2_distribution.rsample()
                classifiers.append([fc2_w, self.bias])

            outputs = []
            for classifier in classifiers:
                out = F.linear(x, classifier[0], classifier[1])
                outputs.append(out)
            return outputs
        else:
            if only_mu:
                out = F.linear(x, self.mu2, self.bias)
                return [out]
            else:
                classifiers = []
                for _ in range(self.num_classifiers_test):
                    fc2_w = fc2_distribution.rsample()
                    classifiers.append([fc2_w, self.bias])

                outputs = []
                for classifier in classifiers:
                    out = F.linear(x, classifier[0], classifier[1])
                    outputs.append(out)
                return outputs



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
    def __init__(self, output_dim, num_classifiers_train=2, num_classifiers_test=20, init='kaiming_u', use_init=False):
        super(CLDNN_C, self).__init__()
        self.num_classifiers_train = num_classifiers_train
        self.num_classifiers_test = num_classifiers_test
        self.init = init

        function_init = {
            'kaiming_u': nn.init.kaiming_uniform_,
            'kaiming_n': nn.init.kaiming_normal_,
            'xavier': nn.init.xavier_normal_
        }

        self.fc1 = nn.Linear(508 * 64, 128)
        self.bn1_fc = nn.BatchNorm1d(128)

        self.mu2 = nn.Parameter(torch.randn(output_dim, 128))
        self.sigma2 = nn.Parameter(torch.zeros(output_dim, 128))

        if use_init:
            all_parameters = [self.mu2, self.sigma2]
            for item in all_parameters:
                function_init[self.init](item)

        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, only_mu=True):
        x = self.fc1(x)
        x = F.relu(self.bn1_fc(x))

        sigma2_pos = torch.sigmoid(self.sigma2)
        fc2_distribution = torch.distributions.Normal(self.mu2, sigma2_pos)

        if self.training:
            classifiers = []
            for _ in range(self.num_classifiers_train):
                fc2_w = fc2_distribution.rsample()
                classifiers.append([fc2_w, self.b2])

            outputs = []
            for classifier in classifiers:
                out = F.linear(x, classifier[0], classifier[1])
                outputs.append(out)

            return outputs
        else:
            if only_mu:
                out = F.linear(x, self.mu2, self.b2)
                return [out]
            else:
                classifiers = []
                for _ in range(self.num_classifiers_test):
                    fc2_w = fc2_distribution.rsample()
                    classifiers.append([fc2_w, self.b2])

                outputs = []
                for classifier in classifiers:
                    out = F.linear(x, classifier[0], classifier[1])
                    outputs.append(out)
                return outputs


class Star:   
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

    def discrepancy_loss(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

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
                outputs_list = classifier(features, only_mu=True)
                outputs = outputs_list[0]
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
        classifier = self.C(output_dim=self.num_classes).to(self.device)
        
        # Define criterion and optimizers
        criterion = nn.CrossEntropyLoss()
        optimizer_g = optim.Adam(feature_extractor.parameters(), lr=self.learning_rate)
        optimizer_c = optim.Adam(classifier.parameters(), lr=self.learning_rate)
        
        # Learning rate schedulers (optional)
        scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
        scheduler_c = optim.lr_scheduler.StepLR(optimizer_c, step_size=10, gamma=0.1)
        
        for epoch in range(self.n_epochs):
            feature_extractor.train()
            classifier.train()
            
            running_loss_s = 0.0
            running_loss_dis = 0.0
            
            source_iter = iter(self.S_train_loader)
            target_iter = iter(self.T_train_loader)
            num_batches = min(len(self.S_train_loader), len(self.T_train_loader))
            
            for batch_idx in range(num_batches):
                ##############################
                # Step 1: Update G and C using source data
                ##############################
                # Unfreeze all parameters
                for param in feature_extractor.parameters():
                    param.requires_grad = True
                for param in classifier.parameters():
                    param.requires_grad = True
                
                optimizer_g.zero_grad()
                optimizer_c.zero_grad()
                
                # Get source batch
                inputs_s, labels_s = next(source_iter)
                inputs_s, labels_s = inputs_s.to(self.device), labels_s.to(self.device)
                
                # Forward pass
                features_s = feature_extractor(inputs_s)
                outputs_s_list = classifier(features_s)
                
                # Compute classification loss on source data
                loss_s = 0
                for outputs_s in outputs_s_list:
                    loss_s += criterion(outputs_s, labels_s)
                loss_s /= len(outputs_s_list)
                
                # Backward and optimize
                loss_s.backward()
                optimizer_g.step()
                optimizer_c.step()
                
                ##############################
                # Step 2: Update classifiers C using target data to maximize discrepancy
                ##############################
                # Freeze feature extractor
                for param in feature_extractor.parameters():
                    param.requires_grad = False
                # Unfreeze classifier
                for param in classifier.parameters():
                    param.requires_grad = True
                
                optimizer_c.zero_grad()
                
                # Get target batch
                inputs_t, _ = next(target_iter)
                inputs_t = inputs_t.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    features_t = feature_extractor(inputs_t)
                outputs_t_list = classifier(features_t)
                
                # Compute discrepancy loss between classifiers
                loss_dis = 0
                num_classifiers = classifier.num_classifiers_train
                for i in range(num_classifiers):
                    for j in range(i + 1, num_classifiers):
                        loss_dis += self.discrepancy_loss(outputs_t_list[i], outputs_t_list[j])
                num_pairs = num_classifiers * (num_classifiers - 1) / 2
                loss_dis = loss_dis / num_pairs
                
                # Maximize discrepancy by minimizing negative loss
                loss_dis = -loss_dis
                loss_dis.backward()
                optimizer_c.step()
                
                ##############################
                # Step 3: Update generator G using target data to minimize discrepancy
                ##############################
                # Unfreeze feature extractor
                for param in feature_extractor.parameters():
                    param.requires_grad = True
                # Freeze classifier
                for param in classifier.parameters():
                    param.requires_grad = False
                
                optimizer_g.zero_grad()
                
                # Forward pass
                features_t = feature_extractor(inputs_t)
                outputs_t_list = classifier(features_t)
                
                # Compute discrepancy loss between classifiers
                loss_dis = 0
                for i in range(num_classifiers):
                    for j in range(i + 1, num_classifiers):
                        loss_dis += self.discrepancy_loss(outputs_t_list[i], outputs_t_list[j])
                loss_dis = loss_dis / num_pairs
                
                # Minimize discrepancy
                loss_dis.backward()
                optimizer_g.step()
                
                ##############################
                # Reset requires_grad for next iteration
                ##############################
                # Unfreeze all parameters for next iteration
                for param in feature_extractor.parameters():
                    param.requires_grad = True
                for param in classifier.parameters():
                    param.requires_grad = True
                
                # Update running losses
                running_loss_s += loss_s.item()
                running_loss_dis += loss_dis.item()
            
            # Learning rate scheduler step
            scheduler_g.step()
            scheduler_c.step()
            
            # Print average losses for the epoch
            avg_loss_s = running_loss_s / num_batches
            avg_loss_dis = running_loss_dis / num_batches
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Class Loss: {avg_loss_s:.4f}, Discrepancy Loss: {avg_loss_dis:.4f}')
            
            # Early stopping or validation steps can be added here if necessary
            
        return feature_extractor, classifier

    # Run multiple times and collect performance metrics
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