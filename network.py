import numpy as np
from functions import sigmoid, sigmoid_derivative, softmax


class FullyConnectedNeuralNetwork:
    def __init__(self, layers):
        """
        Initialize the neural network with a given structure.

        :param layers: List where each element represents the number of neurons in a corresponding layer.
        Example: [3, 5, 2] represents a network with 3 input neurons, 5 in a hidden layer, and 2 output neurons.
        """
        self.layers = layers
        self.weights, self.biases = dict(), dict()

        def xavier_initialization(n_in, n_out):
            return np.sqrt(6 / (n_in + n_out))

        for l in range(1, len(self.layers)):
            self.weights[f'W{l}'] = np.random.normal(
                - xavier_initialization(l, l -1),
                xavier_initialization(l, l - 1),
                size=(self.layers[l], self.layers[l - 1]))

            self.biases[f'b{l}'] = np.random.normal(
                - xavier_initialization(l, 1),
                xavier_initialization(l, 1),
                size=(self.layers[l], 1))


        self.losses = list()

    @staticmethod
    def cross_entropy_loss(A, Y):
        """
        Computes the cross-entropy loss.

        :param A: Predictions from the network.
        :param Y: True labels in one-hot encoded format.
        :return: Scalar loss value.
        """
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A + 1e-9)) / m  # Add small epsilon to avoid log(0)
        return loss

    def forward(self, X):
        """
        Perform forward propagation through the network.

        :param X: Input data (features).
        :return: Dictionary of linear and activation values for each layer.
        """
        forwarding = {'A0': X}

        for l in range(1, len(self.layers)):
            W = self.weights[f'W{l}']
            b = self.biases[f'b{l}']

            # Linear step: Z = W*A + b
            Z = np.dot(W, forwarding[f'A{l - 1}']) + b
            forwarding[f'Z{l}'] = Z

            # Activation step: Apply ReLU for hidden layers and softmax for the output layer
            forwarding[f'A{l}'] = sigmoid(Z) if l < len(self.layers) - 1 else softmax(Z)

        return forwarding

    def backward(self, forwarding, Y):
        """
        Perform backward propagation to compute gradients.

        :param forwarding: Dictionary of linear and activation values for each layer.
        :param Y: True labels in one-hot encoded format.
        :return: Dictionary of gradients for weights and biases.
        """
        gradients = dict()
        m = Y.shape[1]
        L = len(self.layers) - 1  # Number of layers

        # Calculate the gradient for the output layer
        dZ = forwarding[f'A{L}'] - Y
        gradients[f'dW{L}'] = np.dot(dZ, forwarding[f'A{L - 1}'].T) / m
        gradients[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True) / m

        # Back propagate through the hidden layers
        for l in range(L - 1, 0, -1):
            dA = np.dot(self.weights[f'W{l + 1}'].T, dZ)
            dZ = dA * sigmoid_derivative(forwarding[f'Z{l}'])
            gradients[f'dW{l}'] = np.dot(dZ, forwarding[f'A{l - 1}'].T) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        Update weights and biases using the computed gradients.

        :param gradients: Dictionary containing gradients for weights and biases.
        :param learning_rate: Learning rate for gradient descent.
        """
        for l in range(1, len(self.layers)):
            self.weights[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.biases[f'b{l}'] -= learning_rate * gradients[f'db{l}']

    def fit(self, X, Y, epochs=1000, learning_rate=0.01):
        """
        Train the neural network using the given data.

        :param X: Input data (features).
        :param Y: True labels in one-hot encoded format.
        :param epochs: Number of iterations.
        :param learning_rate: Learning rate for gradient descent.
        """
        for epoch in range(epochs):
            # Forward propagation
            cache = self.forward(X)

            # Compute loss
            loss = self.cross_entropy_loss(cache[f'A{len(self.layers) - 1}'], Y)
            self.losses.append(loss)

            # Backward propagation
            gradients = self.backward(cache, Y)

            # Update parameters
            self.update_parameters(gradients, learning_rate)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch} - Loss: {loss}')

    def predict(self, X):
        """
        Predict the class labels for the given input.

        :param X: Input data.
        :return: Predicted class labels.
        """
        cache = self.forward(X)
        output = cache[f'A{len(self.layers) - 1}']
        return np.argmax(output, axis=0)


# # Synthetic dataset generation
# np.random.seed(42)
# m = 200  # Number of samples
# X = np.random.randn(2, m)  # 2D features for each sample
# Y = np.zeros((1, m))
# Y[0, X[0, :] + X[1, :] > 0] = 1  # Simple linear separation: label 1 if sum of features is positive
# Y_one_hot = np.eye(2)[Y.astype(int)].reshape(2, m)  # One-hot encoded labels
#
# # Define and train the neural network
# nn = FullyConnectedNeuralNetwork([2, 4, 2])  # 2 inputs, 4 neurons in the hidden layer, 2 output classes
# nn.fit(X, Y_one_hot, epochs=10000, learning_rate=0.1)
#
# # Predictions
# predictions = nn.predict(X)
# print(f"Predicted labels: {predictions}")
# print(f"True labels: {Y.flatten()}")
