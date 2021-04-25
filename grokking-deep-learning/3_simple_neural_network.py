def neural_network(input_value: float, weight: float) -> float:
    return input_value * weight


def main():
    weight = 0.1
    number_of_toes = [8.5, 9.5, 10, 9]
    input_value = number_of_toes[0]
    prediction = neural_network(input_value, weight)
    print(prediction)


if __name__ == '__main__':
    main()
