def retrieve_data_from(resource):
    """
    Retrieves input pairs and results from a resource/.
    :param resource: file in the resource directory
    :return: a dict containing lists of input pairs and a list of results
    """

    xs = []
    ys = []
    results = []

    with open(resource) as f:
        for line in f.readlines():
            x, y, result = line.strip().split(',')

            xs.append(float(x))
            ys.append(float(y))
            results.append(float(result))

    return {
        'xs': xs,
        'ys': ys,
        'results': results
    }


if __name__ == '__main__':
    a = retrieve_data_from('data/classification2.txt')

    print(a)
    print(a['xs'])
    print(a['ys'])
    print(a['results'])