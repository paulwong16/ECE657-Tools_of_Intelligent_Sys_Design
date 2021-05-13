import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pickle import dump


def create_dataset(path, n_steps=3):
    df = pd.read_csv(path)
    features_column = [" Volume", " Open", " High", " Low"]
    dates = df["Date"].values
    sequence = df[features_column].values
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i+1:end_ix+1], sequence[i][1]
        X.append(seq_x.reshape(12))
        date = dates[i] if len(dates[i]) == 8 else (dates[i][:6] + dates[i][8:])
        y.append([seq_y, date])
    X = np.array(X)
    y = np.array(y)
    n = len(X)
    indices = np.random.permutation(n)
    X = X[indices]
    y = y[indices]
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    columns = ["f" + str(i) for i in range(n_steps*4)]
    columns += ["target", "date"]
    train_df = pd.DataFrame(data=np.concatenate((X_train, y_train), axis=1), columns=columns)
    train_df.to_csv('./data/train_data_RNN.csv')
    test_df = pd.DataFrame(data=np.concatenate((X_test, y_test), axis=1), columns=columns)
    test_df.to_csv('./data/test_data_RNN.csv')


class LSTM(nn.Module):
    def __init__(self, input, hidden, output=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input
        self.hidden_size = hidden
        self.output_size = output
        self.num_layers = num_layers

        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x, _ = self.lstm_layer(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x


if __name__ == "__main__": 
    # create_dataset('./data/q2_dataset.csv')
    train_set = pd.read_csv('./data/train_data_RNN.csv')
    feature_num = len(train_set.columns)
    columns = ["f" + str(i) for i in range(feature_num-3)]
    X_train, y_train = train_set[columns].values, train_set['target'].values.reshape((-1, 1))
    dates = train_set['date'].values.reshape((-1, 1))

    X_sclaer = MinMaxScaler()
    X_train_scaled = X_sclaer.fit_transform(X_train)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = LSTM(input=12, hidden=24, output=1, num_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = model.to(device)

    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_losses = []

    for epoch in range(1, 301):
        output = model(torch.from_numpy(X_train_scaled.reshape(-1, 1, 12)).float().to(device))
        optimizer.zero_grad()
        loss = criterion(output, torch.from_numpy(y_train_scaled.reshape(-1, 1, 1)).float().to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print('Train Epoch: %i, Loss: %f' % (epoch, loss.item()))

    y_pred = model(torch.from_numpy(X_train_scaled.reshape(-1, 1, 12)).float().to(device))
    y_pred_np = y_pred.cpu().detach().numpy().reshape(-1, 1)
    y_pred_np = y_scaler.inverse_transform(y_pred_np)
    columns = ["gt", "pred", "date"]
    df = pd.DataFrame(data=np.concatenate((y_train, y_pred_np, dates), axis=1), columns=columns)
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%y")
    df = df.sort_values(by='date')
    plt.figure()
    plt.plot(df['date'], df['gt'])
    plt.plot(df['date'], df['pred'])
    plt.xlabel('Date')
    plt.ylabel('Open')
    plt.legend(['groundtruth', 'prediction'])
    plt.show()
    plt.figure()
    plt.plot([i for i in range(1, 301)], train_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    save_model = {'model': model, 'x_scaler': X_sclaer, 'y_scaler': y_scaler}
    dump(save_model, open('./models/20856733_RNN_model.pkl', 'wb'))
    pass