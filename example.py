from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from network import FullyConnectedNeuralNetwork
import numpy as np


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
nn = FullyConnectedNeuralNetwork([4, 5, 3])
nn.fit(X_train, Y_train)

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
