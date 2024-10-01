from functions import *


class NeuralNetwork:

    def __init__(self, layers: list, learning_rate=0.01):
        self.layers = layers
        self.parameters = {}

        for l in range(1, len(layers)):
            self.parameters[f'W{l}'] = np.random.randn(layers[l-1], layers[l]) * 0.01
            self.parameters[f'b{l}'] = np.zeros((1, layers[l]))

        self.learning_rate = learning_rate
        self.losses = []

    def forward(self, X):
        for l in range(1, len(self.layers)):

            if l == 1:
                self.parameters[f'Z{l}'] = np.dot(X, self.parameters[f'W{l}']) + self.parameters[f'b{l}']
                self.parameters[f'A{l}'] = sigmoid(self.parameters[f'Z{l}'])
            else:
                self.parameters[f'Z{l}'] = np.dot(self.parameters[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'b{l}']
                self.parameters[f'A{l}'] = sigmoid(self.parameters[f'Z{l}']) \
                    if l < len(self.layers) - 1 else softmax(self.parameters[f'Z{l}'])

        return self.parameters[f'A{len(self.layers) - 1}']

    @staticmethod
    def compute_loss(Y, Y_hat):
        # Cross-entropy loss
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[range(m), Y.argmax(axis=1)])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y, Y_hat):
        # Backward propagation
        m = X.shape[0]

        for l in range(1, len(self.layers))[::-1]:
            if l == len(self.layers) - 1:
                self.parameters[f'dZ{l}'] = Y_hat - Y
                self.parameters[f'dW{l}'] = np.dot(self.parameters[f'A{l - 1}'].T, self.parameters[f'dZ{l}']) / m
                self.parameters[f'db{l}'] = np.sum(self.parameters[f'dZ{l}'], axis=0, keepdims=True) / m
            else:
                self.parameters[f'dA{l}'] = np.dot(self.parameters[f'dZ{l + 1}'], self.parameters[f'W{l + 1}'].T)
                self.parameters[f'dZ{l}'] = self.parameters[f'dA{l}'] * sigmoid_derivative(self.parameters[f'Z{l}'])
                self.parameters[f'dW{l}'] = np.dot(self.parameters[f'A{l - 1}'].T, self.parameters[f'dZ{l}']) / m \
                    if l > 1 else np.dot(X.T, self.parameters[f'dZ{l}']) / m
                self.parameters[f'db{l}'] = np.sum(self.parameters[f'dZ{l}'], axis=0, keepdims=True) / m

            self.parameters[f'W{l}'] -= self.learning_rate * self.parameters[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.parameters[f'db{l}']

    def train(self, X, Y, epochs=1000):
        # Train the network
        for i in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.losses.append(loss)
            self.backward(X, Y, Y_hat)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)
