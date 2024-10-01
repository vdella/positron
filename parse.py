import numpy as np


def retrieve_data_from(resource):
    """
    Retrieves input pairs and results from a resource/.
    :param resource: file in the resource directory
    :return: a dict containing np.arrays of input pairs and a np.array of results
    """

    inputs = []
    results = []

    with open(resource) as f:
        for line in f.readlines():
            x, y, result = line.strip().split(',')

            inputs.append([float(x), float(y)])
            results.append([float(result)])

    return {
        'inputs': np.array(inputs),
        'results': np.array(results)
    }


if __name__ == '__main__':
    a = retrieve_data_from('data/classification2.txt')

    print(a['inputs'])
    print(a['results'])