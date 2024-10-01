import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, sigmoid_derivative, softmax

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.losses = []

    def forward(self, X):
        # Forward propagation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y, Y_hat):
        # Cross-entropy loss
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[range(m), Y.argmax(axis=1)])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y, Y_hat):
        # Backward propagation
        m = X.shape[0]
        dZ2 = Y_hat - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

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

# Load the Iris dataset
data = load_iris()
X = data.data  # 4 features (sepal length, sepal width, petal length, petal width)
Y = data.target.reshape(-1, 1)  # 3 classes (Setosa, Versicolour, Virginica)

# One-hot encode the target values
encoder = OneHotEncoder(sparse_output=False)
Y_encoded = encoder.fit_transform(Y)

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# Initialize and train the neural network
nn = NeuralNetwork(input_size=4, hidden_size=4, output_size=3, learning_rate=0.1)
nn.train(X_train, Y_train, epochs=1000)

# Plot the loss over epochs
plt.plot(nn.losses)
plt.title("Loss over Epochs for the Iris Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Predict on the test set and calculate accuracy
predictions = nn.predict(X_test)
accuracy = np.mean(np.argmax(Y_test, axis=1) == predictions)
print(f"Test Accuracy: {accuracy:.4f}")
