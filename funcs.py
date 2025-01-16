import numpy as np
import torch
import h5py

from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_loader(X_path, Y_path, batch_size=80, split=0.8, permute=True):
    """
    Turn NPY files into dataloaders. Input can be a file path or an ndarray.

    Parameters
    ----------
    X_path : Path for feature set
    Y_path : Path for label set 
    batch_size : Batch size of dataloader
    split : Split used for train and validation

    Returns
    -------
    train_loader : Train dataloader
    val_loader : Test dataloader

    """
    if isinstance(X_path, np.ndarray):
        X = X_path
    else :
        X = np.load(X_path)
    
    if isinstance(Y_path, np.ndarray):
        Y = Y_path
    else:
        Y = np.load(Y_path)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)
    
    if permute:
        X_tensor = X_tensor.permute(0, 2, 1)
        
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def filter_and_save_data(file_path, all_classes, class_of_interest, z_value, domain):
    """
    Imported h5py dataset is converted into sampled NPY files given a class
    of interest (COI). COI are the modulations subset.

    Parameters
    ----------
    file_path : Path to h5py dataset
    all_classes : Set of all classes
    class_of_interest : Subset of modulations we want to filter
    z_value : Value of SNR we want to filter for
    domain : Source or target

    Returns
    -------
    Saved .npy files in ./data/

    """
    # Map class names to indices
    coi_indices = [all_classes.index(name) for name in class_of_interest]

    with h5py.File(file_path, 'r') as f:
        X = f['X'][:] # Feature set
        Y = f['Y'][:] # Label
        Z = f['Z'][:] # SNR 

    # Filter data to include only specified classes and Z value
    selected_indices = [
        i for i, (label, z_val) in enumerate(zip(np.argmax(Y, axis=1), Z))
        if label in coi_indices and z_val == z_value
    ]
    X_selected = X[selected_indices]
    Y_selected = Y[selected_indices]
    Z_selected = Z[selected_indices]

    # Convert one-hot labels to categorical labels
    templabels = np.argmax(Y_selected, axis=1)
    unique_labels = np.unique(templabels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    Y_selected = np.array([label_map[label] for label in templabels])

    np.save(f'data/{domain}_X.npy', X_selected)
    np.save(f'data/{domain}_Y.npy', Y_selected)
    np.save(f'data/{domain}_Z.npy', Z_selected)

    print('Files saved in /data/*.npy')
    return f'data/{domain}_X.npy', f'data/{domain}_Y.npy'