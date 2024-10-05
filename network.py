from functions import hidden_activation, output_activation
import numpy as np


class NeuralNetwork:

    def __init__(self, layers: list, learning_rate=0.01, hidden_act='sigmoid', output_act='sigmoid'):
        """Standard fully connected neural network initialization.

        :param layers: list of integers, where each integer represents the number of neurons in each layer.
        :param learning_rate: float, the learning rate of the network.
        :param hidden_act: string, the activation function to be used in the hidden layers.
        :param output_act: string, the activation function to be used in the output layer.
        """

        self.layers = layers
        self.parameters = {}

        for l in range(1, len(layers)):
            self.parameters[f'W{l}'] = np.random.randn(layers[l - 1], layers[l]) * np.sqrt(6.0 / layers[l - 1])
            self.parameters[f'b{l}'] = np.zeros((1, layers[l]))

        self.hidden_act_name = hidden_act
        self.hidden_act_function = hidden_activation[hidden_act][0]
        self.hidden_act_derivative_function = hidden_activation[hidden_act][1]

        self.output_act = output_activation[output_act]

        self.learning_rate = learning_rate
        self.losses = {}  # Records losses along the epochs in order to plot them later.

    def forward(self, X):
        self.parameters[f'Z1'] = (np.dot(X, self.parameters[f'W1'])
                                  + self.parameters[f'b1'])
        self.parameters[f'A1'] = self.hidden_act_function(self.parameters[f'Z1'])

        for l in range(2, len(self.layers)):
            self.parameters[f'Z{l}'] = (np.dot(
                self.parameters[f'A{l-1}'], self.parameters[f'W{l}'])
                                        + self.parameters[f'b{l}'])
            self.parameters[f'A{l}'] = self.hidden_act_function(self.parameters[f'Z{l}'])

        return self.parameters[f'A{len(self.layers) - 1}']

    @staticmethod
    def compute_loss(Y, Y_hat):
        """
        Computes the cross-entropy loss between the true labels and predicted probabilities.

        :param Y: numpy array containing the true labels.
        :param Y_hat: numpy array containing the predicted probabilities.
        :return: Cross-entropy loss value.
        """
        # Clip the predicted values to a small value in order to avoid log(0) errors.
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # Compute the cross-entropy loss
        loss = -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

        return loss

    def backward(self, X, Y, Y_hat):
        for l in range(1, len(self.layers))[::-1]:
            if l == len(self.layers) - 1:
                self.parameters[f'dZ{l}'] = Y_hat - Y
                self.parameters[f'dW{l}'] = np.dot(self.parameters[f'A{l - 1}'].T,
                                                   self.parameters[f'dZ{l}'])
                self.parameters[f'db{l}'] = np.sum(self.parameters[f'dZ{l}'],
                                                   axis=0, keepdims=True)
            else:
                self.parameters[f'dA{l}'] = np.dot(self.parameters[f'dZ{l + 1}'],
                                                   self.parameters[f'W{l + 1}'].T)
                self.parameters[f'dZ{l}'] = (self.parameters[f'dA{l}']
                                             * self.hidden_act_derivative_function(self.parameters[f'Z{l}']))
                self.parameters[f'dW{l}'] = np.dot(self.parameters[f'A{l - 1}'].T,
                                                   self.parameters[f'dZ{l}']) \
                    if l > 1 else np.dot(X.T, self.parameters[f'dZ{l}'])
                self.parameters[f'db{l}'] = np.sum(self.parameters[f'dZ{l}'],
                                                   axis=0,
                                                   keepdims=True)

            self.parameters[f'W{l}'] -= self.learning_rate * self.parameters[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.parameters[f'db{l}']

    def train(self, X, Y, epochs=1000):

        for i in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.losses[i] = loss
            self.backward(X, Y, Y_hat)

    def predict(self, X):
        Y_hat = self.forward(X)
        if self.hidden_act_name == 'sigmoid':
            return (Y_hat >= 0.5).astype(int)  # Threshold for binary classification
        else:
            return np.argmax(Y_hat, axis=1).reshape(-1, 1)
