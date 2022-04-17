import numpy as np

np.random.seed(1)


# Rectified Linear Unit
# Это наиболее часто используемая функция активации при глубоком обучении.
# Данная функция возвращает 0, если принимает отрицательный аргумент,
# в случае же положительного аргумента, функция возвращает само число.
def relu(x: np.ndarray):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


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
    print("\n")

    for i in range(60):
        layer_2_error = 0
        for j in range(len(street_lights)):
            print(str(i) + " / " + str(j) + ":")
            layer_0 = street_lights[j:j+1]
            print("layer_0 = " + str(layer_0))
            print("Dot product layer_0 and weights_0_1 = " + str(np.dot(layer_0, weights_0_1)))
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            print("layer_1 = " + str(layer_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            print("layer_2 = " + str(layer_2))

            layer_2_error += np.sum((layer_2 - walk_or_stop[j:j+1]) ** 2)
            print("layer_2_error = " + str(layer_2_error))

            layer_2_delta = (walk_or_stop[j:j+1] - layer_2)
            print("layer_2_delta = " + str(layer_2_delta))
            # https://stackoverflow.com/questions/39608421/showing-valueerror-shapes-1-3-and-1-3-not-aligned-3-dim-1-1-dim-0
            layer_1_delta = layer_2_delta.dot(np.squeeze(np.asarray(weights_1_2.T))) * relu2deriv(layer_1)
            print("layer_1_delta = " + str(layer_1_delta))

            weights_1_2 = alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 = alpha * layer_0.T.dot(layer_1_delta)
            print("weights_1_2 = " + str(weights_1_2))
            print("weights_0_1 = " + str(weights_0_1))

            print("\n")


if __name__ == '__main__':
    main()




