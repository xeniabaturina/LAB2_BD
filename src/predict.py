import torch
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.utils import load_model
import configparser


def predict(test_path, model_path, scaler_path, predictions_path):
    _, _, x_test, ids = load_and_preprocess_data(None, test_path, scaler_path=scaler_path)
    model = load_model(model_path, input_size=x_test.shape[1])

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        test_predictions = model(x_test_tensor)

    submission_df = pd.DataFrame({
        'ID': ids,
        'medv': test_predictions.numpy().flatten()
    })

    submission_df.to_csv(predictions_path, index=False)
    print(f'Submission saved to {predictions_path}')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_test = '../data/test.csv'
    path_to_model = config['model']['model_path']
    path_to_scaler = config['preprocessing']['scaler_path']
    path_to_predictions = 'submission_my_model.csv'
    predict(path_to_test, path_to_model, path_to_scaler, path_to_predictions)
