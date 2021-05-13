import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time as timer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_nodes=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, 10)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_size, hidden_nodes=None):
        super(MLP2, self).__init__()
        if hidden_nodes is None:
            hidden_nodes = [1024, 512]
        self.fc1 = nn.Linear(input_size, hidden_nodes[0])
        self.fc2 = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        self.fc3 = nn.Linear(hidden_nodes[1], 10)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP3(nn.Module):
    def __init__(self, input_size, hidden_nodes=None):
        super(MLP3, self).__init__()
        if hidden_nodes is None:
            hidden_nodes = [1024, 512, 512]
        self.fc1 = nn.Linear(input_size, hidden_nodes[0])
        self.fc2 = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        self.fc3 = nn.Linear(hidden_nodes[1], hidden_nodes[2])
        self.fc4 = nn.Linear(hidden_nodes[2], 10)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*32*32, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*32*32)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 64*8*8)
        x = torch.sigmoid(self.fc1(x))
        x = F.dropout(x, 0.2)
        x = torch.sigmoid(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x


def train(model, optimizer, criterion, train_loader, val_loader=None, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(1, epochs+1):
        start = timer.time()
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
        accuracy = correct / len(train_loader.dataset)
        train_accuracies.append(accuracy)
        print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch, train_loss, accuracy))
        print('Training time for this epoch %f' % (timer.time() - start))
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    loss = criterion(output, target)
                    val_loss += loss.item()
            val_losses.append(val_loss)
            accuracy = correct / len(val_loader.dataset)
            val_accuracies.append(accuracy)
            print('Val Epoch:   %i, Loss: %f, Accuracy: %f' % (epoch, val_loss, accuracy))
    return train_losses, train_accuracies, val_losses, val_accuracies


def predict(model, criterion, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            test_loss += loss.item()
    accuracy = correct / len(test_loader.dataset)
    print('Test Loss: %f, Test Accuracy: %f' % (test_loss, accuracy))


if __name__ == '__main__':
    train_set = datasets.CIFAR10(root='./data/', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    val_set = datasets.CIFAR10(root='./data/', train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    n_train = len(train_set)
    batch_size = 32
    split = n_train // 5
    indices = np.random.permutation(n_train)
    subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=subset_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = train_set.data[0].reshape((-1, 1)).shape[0]
    model = MLP(input_size=input_size)
    # model = CNN1()
    # model = CNN2()
    # config
    model = model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, 6):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        accuracy = correct / len(train_loader.dataset)
        train_accuracies.append(accuracy)
        print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch, train_loss, accuracy))
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / len(val_loader.dataset)
        val_accuracies.append(accuracy)
        print('Val Epoch:   %i, Loss: %f, Accuracy: %f' % (epoch, val_loss, accuracy))