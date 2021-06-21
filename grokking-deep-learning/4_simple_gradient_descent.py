def neural_network( input_value: float, weight: float) -> float:
    return weight * input_value


def get_error(prediction: float, real: float) -> float:
    return (prediction - real) ** 2


def main():
    weight = 0.5
    input_value = 0.5
    real = 0.8

    for i in range(20):
        prediction = neural_network(input_value, weight)
        mead_squared_error = get_error(prediction, real)
        corr = (prediction - real) * input_value
        weight = weight - corr

        print(
            "Error: " + str(mead_squared_error) +
            " Prediction: " + str(prediction) +
            " Correction: " + str(corr) +
            " Weight: " + str(weight)
        )


if __name__ == '__main__':
    main()
