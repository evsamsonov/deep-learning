import numpy as np

np.random.seed(1)


def relu(x: np.ndarray):
    return (x > 0) * x


def main():
    alpha = 0.2
    hidden_size = 4     # Размер скрытого слоя

    street_lights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ])

    walk_or_stop = np.array([
        [1],
        [1],
        [0],
        [0],
    ]).T

    # Генерируем веса случайным образом
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1    # 3 ряда по 4 элемента
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1    # 4 ряда по 1 элементу

    print("weights_0_1 = " + str(weights_0_1))
    print("weights_1_2 = " + str(weights_1_2))

    layer_0 = street_lights[0]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    print("layer_1 = " + str(layer_1))
    layer_2 = np.dot(layer_1, weights_1_2)
    print("layer_2 = " + str(layer_2))


if __name__ == '__main__':
    main()




