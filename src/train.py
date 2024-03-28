import torch.optim as optim
import torch.nn as nn
from .preprocess import load_and_preprocess_data, create_dataloaders
from .utils import RegressionModel, save_model
import configparser


def train(train_path, model_path, scaler_path, num_epochs=600, lr=0.001):
    x_train, y_train, _, _ = load_and_preprocess_data(train_path, None, scaler_path=scaler_path)
    train_loader = create_dataloaders(x_train, y_train)

    model = RegressionModel(x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_train = config['data']['train_data_path']
    path_to_model = config['model']['model_path']
    path_to_scaler = config['preprocessing']['scaler_path']
    epochs_number = int(config['hyperparameters']['num_epochs'])
    learning_rate = float(config['hyperparameters']['learning_rate'])

    train(path_to_train, path_to_model, path_to_scaler, epochs_number, learning_rate)
