fn main() {
    let weight = 0.1;
    let number_of_toes: [f64; 4] = [8.5, 9.5, 10., 9.];
    let input = number_of_toes[0];

    let prediction = neural_network(input, weight);
    println!("{}", prediction)
}

fn neural_network(input: f64, weight: f64) -> f64 {
    return input * weight
}