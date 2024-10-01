def parse(resource):
    input_feature_pair = []
    results = []

    with open(resource) as f:
        for line in f.readlines():
            x, y, result = line.strip().split(',')
            input_feature_pair.append((float(x), float(y)))
            results.append(float(result))

    return input_feature_pair, results


if __name__ == '__main__':
    a = parse('resources/classification2.txt')
    print(a)