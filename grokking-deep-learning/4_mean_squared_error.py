def main():
    weight = 0.5
    input_value = 0.5
    real = 0.8

    prediction = weight * input_value

    mead_squared_error = (prediction - real) ** 2
    print(mead_squared_error)


if __name__ == '__main__':
    main()
