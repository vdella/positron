import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from network import NeuralNetwork


def normalize(arr):
    return (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))


def calculate_metrics(output_test, predictions):
    tp = np.sum((predictions == 1) & (output_test == 1))  # True Positives
    fp = np.sum((predictions == 1) & (output_test == 0))  # False Positives
    fn = np.sum((predictions == 0) & (output_test == 1))  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def plot_results(X, y, nn, h=0.01):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Create subplots side by side

    # Set the range of values for the grid for the decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Flatten the grid to pass it through the model for predictions
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict the class labels (use probabilities if available)
    Z = nn.predict(grid)
    Z = Z.reshape(xx.shape)

    # Create the decision boundary plot
    ax1.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')  # Decision boundary line
    ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Spectral)
    ax1.set_title("Decision Boundary")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    # Plot the loss over epochs on the second subplot
    ax2.plot(list(nn.losses.keys()), list(nn.losses.values()), label='Loss')
    ax2.set_title("Loss over epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    # Show both plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)

    dataset = 'data/classification2.txt'

    data = pd.read_csv(dataset, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    normalize = lambda a: (a - np.min(a, axis=0)) / (np.max(a, axis=0)
                                                     - np.min(a, axis=0))
    X = normalize(X)

    input_train, input_test, output_train, output_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nn = NeuralNetwork([2, 20, 20, 1], learning_rate=0.001, hidden_act='sigmoid', output_act='sigmoid')
    nn.train(input_train, output_train, epochs=10000)

    for epoch in nn.losses.keys():
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {nn.losses[epoch]:.4f}")

    # Assuming you have a trained model and input data, you can call this function as follows:
    plot_results(X, y, nn)

    predictions = nn.predict(input_test)
    accuracy = np.mean(np.argmax(output_test, axis=1) == predictions)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(output_test, predictions)
    recall = recall_score(output_test, predictions)
    f1 = f1_score(output_test, predictions)

    # Calculate Precision, Recall, and F1 Score using numpy
    precision, recall, f1 = calculate_metrics(output_test, predictions)

    print(f'Precision (Numpy): {precision:.4f}')
    print(f'Recall (Numpy): {recall:.4f}')
    print(f'F1 Score (Numpy): {f1:.4f}')

    # Display metrics
    print(f'Sklearn-precision: {precision:.4f}')
    print(f'Sklearn-recall: {recall:.4f}')
    print(f'Sklearn-F1 Score: {f1:.4f}')
