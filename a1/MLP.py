import numpy as np
import pickle


class MLP:
    # Initialize the network with given hidden layers and activation functions
    def __init__(self, input_shape=1, output_shape=1, hidden_layers=None, activation_functions=None):
        if hidden_layers is None:
            hidden_layers = [1]
        self.layers = [[input_shape, hidden_layers[0]]]
        for i in range(len(hidden_layers) - 1):
            self.layers.append([hidden_layers[i], hidden_layers[i + 1]])
        self.layers.append([hidden_layers[-1], output_shape])
        if activation_functions:
            self.activations = activation_functions
        else:
            self.activations = ['sigmoid'] * (len(self.layers) - 1)

        self.params = {}

        self.memo = {}

        self.gradients = {}

        for idx, layer in enumerate(self.layers):
            layer_idx = idx + 1
            self.params['W' + str(layer_idx)] = np.random.randn(layer[1], layer[0])
            self.params['b' + str(layer_idx)] = np.random.randn(layer[1], 1)

    # Activation functions and their derivatives
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_derivative(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=0))
        sum = np.sum(exps, axis=0, keepdims=True)
        return exps / sum

    def softmax_derivative(self, dA, Z):
        softm = self.softmax(Z)
        return dA * softm * (1 - softm)

    def tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def tanh_derivative(self, dA, Z):
        return dA * (1 - (self.tanh(Z) * self.tanh(Z)))

    # Forward propagation for single perceptron
    def foward_propagation(self, A_prev, W_cur, b_cur, activation="sigmoid"):
        Z_cur = np.dot(W_cur, A_prev) + b_cur

        if activation == "relu":
            return self.relu(Z_cur), Z_cur
        elif activation == "sigmoid":
            return self.sigmoid(Z_cur), Z_cur
        elif activation == "tanh":
            return self.tanh(Z_cur), Z_cur
        elif activation == "softmax":
            return self.softmax(Z_cur), Z_cur

    # Combined forward propagation
    def foward_propagations(self, X):
        A_cur = X
        for idx, leyer in enumerate(self.layers):
            layer_idx = idx + 1
            A_prev = A_cur
            activ = self.activations[idx]
            W_cur = self.params["W" + str(layer_idx)]
            b_cur = self.params["b" + str(layer_idx)]
            A_cur, Z_cur = self.foward_propagation(A_prev, W_cur, b_cur, activ)

            self.memo["A" + str(idx)] = A_prev
            self.memo["Z" + str(layer_idx)] = Z_cur

        return A_cur

    # Backward propagation for single perceptron
    def backward_propagation(self, dA_cur, W_cur, b_cur, Z_cur, A_prev, activation="sigmoid"):
        m = A_prev.shape[1]

        # Chain rule: dL/dA dA/dZ dZ/dW & dZ/db
        # dL/dZ = dL/dA * dA/dZ
        if activation == "relu":
            dZ_cur = self.relu_derivative(dA_cur, Z_cur)
        elif activation == "sigmoid":
            dZ_cur = self.sigmoid_derivative(dA_cur, Z_cur)
        elif activation == "tanh":
            dZ_cur = self.tanh_derivative(dA_cur, Z_cur)
        elif activation == "softmax":
            dZ_cur = self.softmax_derivative(dA_cur, Z_cur)

        # dL/dW = dL/dZ * dZ/dW
        dW_cur = np.dot(dZ_cur, A_prev.T) / m
        # dL/db = dL/dZ * dZ/db
        db_cur = np.sum(dZ_cur, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_cur.T, dZ_cur)

        return dA_prev, dW_cur, db_cur

    # Combined backward propagations
    def backward_propagations(self, Y_hat, Y):
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        # dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        # Derivative of cost function dL/dA
        dA_prev = Y_hat - Y

        for layer_idx_prev, layer in reversed(list(enumerate(self.layers))):
            layer_idx_cur = layer_idx_prev + 1
            activ = self.activations[layer_idx_prev]

            dA_cur = dA_prev

            A_prev = self.memo["A" + str(layer_idx_prev)]
            Z_cur = self.memo["Z" + str(layer_idx_cur)]
            W_cur = self.params["W" + str(layer_idx_cur)]
            b_cur = self.params["b" + str(layer_idx_cur)]

            # Calculate gradients for w, b, A
            dA_prev, dW_cur, db_cur = self.backward_propagation(dA_cur, W_cur, b_cur, Z_cur, A_prev, activ)

            self.gradients["dW" + str(layer_idx_cur)] = dW_cur
            self.gradients["db" + str(layer_idx_cur)] = db_cur

    def update(self, learning_rate):
        for idx, layer in enumerate(self.layers):
            self.params["W" + str(idx+1)] -= learning_rate * self.gradients["dW" + str(idx+1)]
            self.params["b" + str(idx+1)] -= learning_rate * self.gradients["db" + str(idx+1)]

    def cost_function(self, y_hat, y):
        epsilon = 1e-12
        y_hat = y_hat.T
        y = y.T
        # y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        N = y_hat.shape[0]
        ce = -np.sum(y * np.log(y_hat + 1e-9)) / N
        return ce
        # tmp = np.linalg.norm((y_hat - y), ord=2, axis=0)
        # return tmp.mean()

    def get_accuracy(self, y_hat, y):
        n = y_hat.shape[1]
        res = 0
        for i in range(n):
            if (y_hat[:, i] == y[:, i]).all():
                res += 1
        return res/n

    def prob_to_cat(self, y_hat, y):
        res = np.zeros(y_hat.shape)
        y_hat_cat = np.argmax(y_hat, axis=0)
        res[y_hat_cat, np.arange(y_hat_cat.size)] = 1
        return self.get_accuracy(res, y)

    def predict(self, X):
        A_cur = X
        for idx, leyer in enumerate(self.layers):
            layer_idx = idx + 1
            A_prev = A_cur
            activ = self.activations[idx - 1]
            W_cur = self.params["W" + str(layer_idx)]
            b_cur = self.params["b" + str(layer_idx)]
            A_cur, Z_cur = self.foward_propagation(A_prev, W_cur, b_cur, activ)

        y_hat = A_cur

        res = np.zeros(y_hat.shape)
        y_hat_cat = np.argmax(y_hat, axis=0)
        res[y_hat_cat, np.arange(y_hat_cat.size)] = 1

        return res

    def train(self, X, y, X_val, y_val, epochs, learning_rate):
        cost = []
        acc = []
        val_acc = []

        for i in range(epochs):
            y_hat = self.foward_propagations(X)
            cost.append(self.cost_function(y_hat, y))
            acc.append(self.prob_to_cat(y_hat, y))
            val_acc.append(self.get_accuracy(self.predict(X_val), y_val))

            print('Epoch: %i, Loss: %f, Training Accuracy: %f, Validation Accuracy: %f' % (i+1, cost[-1], acc[-1], val_acc[-1]))

            self.backward_propagations(y_hat, y)
            self.update(learning_rate)

        return cost, acc, val_acc

    def save_model(self, path):
        model_path = path + 'pretrained_model.pkl'
        model = {'params': self.params, 'layers': self.layers, 'activations': self.activations}
        with open(model_path, "wb") as fp:
            pickle.dump(model, fp)

    def read_model(self, model_path):
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        self.params = model['params']
        self.layers = model['layers']
        self.activations = model['activations']
