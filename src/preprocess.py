import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import os


def load_and_preprocess_data(train_path, test_path, scaler_path):
    x_train_scaled = y_train = x_test_scaled = ids = None

    if train_path is not None:
        train_df = pd.read_csv(train_path)
        x_train = train_df.drop('medv', axis=1).values
        y_train = train_df['medv'].values

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Load the pre-fitted scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError("Scaler file not found. Ensure training is run first.")

    if test_path is not None:
        test_df = pd.read_csv(test_path)
        x_test = test_df.values
        ids = test_df['ID']

        x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, ids


def create_dataloaders(x_train, y_train, batch_size=64):
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader
