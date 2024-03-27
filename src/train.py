import torch.optim as optim
import torch.nn as nn
from src.preprocess import load_and_preprocess_data, create_dataloaders
from src.utils import RegressionModel, save_model


def train(train_path, model_path, scaler_path, num_epochs=600):
    x_train, y_train, _, _ = load_and_preprocess_data(train_path, None, scaler_path=scaler_path)
    train_loader = create_dataloaders(x_train, y_train)

    model = RegressionModel(x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss = None
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    save_model(model, model_path)


if __name__ == "__main__":
    path_to_train = '../data/train.csv'
    path_to_model = 'regression_model.pth'
    path_to_scaler = 'scaler.pkl'
    train(path_to_train, path_to_model, path_to_scaler)
