from train_RNN import LSTM
from pickle import load
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt


if __name__ == "__main__":
    saved_model = load(open('./models/20856733_RNN_model.pkl', 'rb'))
    model = saved_model['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_scaler = saved_model['x_scaler']
    y_scaler = saved_model['y_scaler']
    test_set = pd.read_csv('./data/test_data_RNN.csv')
    dates = test_set['date'].values.reshape((-1, 1))
    feature_num = len(test_set.columns)
    columns = ["f" + str(i) for i in range(feature_num - 3)]
    X_test, y_test = test_set[columns].values, test_set['target'].values.reshape((-1, 1))
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)
    targets = torch.from_numpy(y_test_scaled.reshape(-1, 1, 1)).float().to(device)
    y_pred = model(torch.from_numpy(X_test_scaled.reshape(-1, 1, 12)).float().to(device))
    loss_func = torch.nn.MSELoss()
    loss = loss_func(y_pred, targets)
    print(loss.item())
    y_pred = y_pred.cpu().detach().numpy().reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred)
    columns = ["gt", "pred", "date"]
    df = pd.DataFrame(data=np.concatenate((y_test, y_pred, dates), axis=1), columns=columns)
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%y")
    df = df.sort_values(by='date')
    plt.figure()
    plt.plot(df['date'], df['gt'])
    plt.plot(df['date'], df['pred'])
    plt.xlabel('Date')
    plt.ylabel('Open')
    plt.legend(['groundtruth', 'prediction'])
    plt.show()
    pass