import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from network import NeuralNetwork
from display import present_losses_over_epochs


if __name__ == '__main__':
    np.random.seed(42)

    iris = load_iris()

    X = iris.data  # 4 features (sepal length, sepal width, petal length, petal width)
    Y = iris.target.reshape(-1, 1)  # 3 classes (Setosa, Versicolour, Virginica)

    # One-hot encode the target values
    encoder = OneHotEncoder(sparse_output=False)
    Y_encoded = encoder.fit_transform(Y)

    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the neural network
    nn = NeuralNetwork([4, 20, 20, 3], learning_rate=0.01, hidden_act='sigmoid', output_act='softmax')
    nn.train(X_train, Y_train, epochs=10000)

    present_losses_over_epochs(nn)

    # Plot the loss over epochs
    plt.plot(nn.losses.keys(), nn.losses.values())
    plt.title('Loss over epochs for the Iris dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Predict on the test set and calculate accuracy
    predictions = nn.predict(X_test)
    accuracy = np.mean(np.argmax(Y_test, axis=1) == predictions)
    print(f"Test Accuracy: {accuracy:.4f}")