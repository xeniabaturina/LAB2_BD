import torch
import pandas as pd
from preprocess import load_and_preprocess_data
from utils import load_model


def predict(test_path, model_path, scaler_path):
    _, _, x_test, ids = load_and_preprocess_data(None, test_path, scaler_path=scaler_path)
    model = load_model(model_path, input_size=x_test.shape[1])

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        test_predictions = model(x_test_tensor)

    submission_df = pd.DataFrame({
        'ID': ids,
        'medv': test_predictions.numpy().flatten()
    })

    submission_file = 'submission_my_model.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f'Submission saved to {submission_file}')


if __name__ == "__main__":
    path_to_test = '../data/test.csv'
    path_to_model = 'regression_model.pth'
    path_to_scaler = 'scaler.pkl'
    predict(path_to_test, path_to_model, path_to_scaler)
