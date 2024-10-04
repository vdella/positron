import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from network import NeuralNetwork


def normalize(arr):
    return (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))


if __name__ == '__main__':
    np.random.seed(42)

    dataset = 'data/classification2.txt'

    data = pd.read_csv(dataset, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    normalize = lambda a: (a - np.min(a, axis=0)) / (np.max(a, axis=0)
                                                     - np.min(a, axis=0))
    X = normalize(X)

    side = np.random.permutation(2)

    input_train = X[side[:1]]
    output_train = y[side[:1]]

    input_test = X[side[1:]]
    output_test = y[side[1:]]

    nn = NeuralNetwork([2, 20, 1], learning_rate=0.0005, hidden_act='relu', output_act='sigmoid')
    nn.train(input_train, output_train, epochs=10000)

    for epoch in nn.losses.keys():
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {nn.losses[epoch]:.4f}")

    plt.plot(nn.losses.keys(), nn.losses.values())
    plt.title("Loss over epochs for the standard dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    predictions = nn.predict(input_test)
    accuracy = np.mean(np.argmax(output_test, axis=1) == predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
