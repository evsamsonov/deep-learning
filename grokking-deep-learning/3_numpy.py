import numpy as np


def main():
    vector1 = np.array([0, 1, 2, 3])
    vector2 = np.array([10, 1, 100, 1000])
    matrix = np.array([
        [0, 1, 2, 3],
        [4, 3, 1, 0],
    ])
    zero_matrix = np.zeros((2, 4))  # Матрица, заполненная нулями - 2 строки, 4 столбца
    rand_matrix = np.random.rand(2, 4)

    print(vector1)
    print(matrix)
    print(zero_matrix)
    print(rand_matrix)

    # Векторно скалярное умножение
    print(vector1 * 2)          # Возвращает вектор, в котором каждый элемент умножается на 2

    # Матрично скалярное умножение
    print(matrix * 2)           # Возвращает  матрицу, в котором каждый элемент умножается на 2

    # Возвращает вектор, поэлементное перемножение векторов - размерность должна совпадать
    # Количество элементов должны совпадать
    print(vector1 * vector2)

    # Возвращает матрицу, поэлементное перемножение вектора на каждую сроку матрицы
    # Количество столбцов вектора и матрицы должны совпадать
    print(vector1 * matrix)

    # Вычисляется скалярные произведения строки матрицы a
    # на каждый столбец матрицы b
    zero_matrix_a = np.array([[1, 2, 3, 4]])  # np.zeros((1, 4))
    zero_matrix_b = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ])        # np.zeros((4, 3))

    zero_matrix_c = zero_matrix_a.dot(zero_matrix_b)
    print(zero_matrix_c)
    print(zero_matrix_c.shape)

    print(zero_matrix_b.T)  # Транспонирование - поменяет местами столбцы и строки


if __name__ == '__main__':
    main()
