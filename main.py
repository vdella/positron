import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from network import NeuralNetwork


def normalize(arr):
    return (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))


if __name__ == '__main__':
    np.random.seed(100)

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

    nn = NeuralNetwork([2, 200, 200, 1], learning_rate=0.0005)
    nn.train(input_train, output_train, epochs=10000)

    plt.plot(nn.losses)
    plt.title("Loss over Epochs for the standard Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('loss-200neurons-200neurons-lr3zero.png')
    plt.show()

    predictions = nn.predict(input_test)
    accuracy = np.mean(np.argmax(output_test, axis=1) == predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
