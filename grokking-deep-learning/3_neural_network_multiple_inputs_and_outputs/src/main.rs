extern crate ndarray;
extern crate blas_src;

use ndarray::{Array, ArrayBase, OwnedRepr, Ix1};

// see https://github.com/rust-ndarray/ndarray
// see https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html
// need install openblas lib

fn main() {
    let game_counts = [8.5, 9.5, 9.9, 9.0];
    let win_shares = [0.65, 0.8, 0.8, 0.9];
    let fan_counts = [1.2, 1.3, 0.5, 1.0];

    let weights = [
        Array::from_vec(vec![0.1, 0.1, -0.3]),
        Array::from_vec(vec![0.1, 0.2, 0.0]),
        Array::from_vec(vec![0.0, 1.3, 0.1])
    ];

    let input = Array::from_vec(vec![game_counts[0], win_shares[0], fan_counts[0]]);
    let prediction = neural_network(input, &weights);
    println!("{:?}", prediction)
}

fn neural_network(input: ArrayBase<OwnedRepr<f64>, Ix1>, weights: &[ArrayBase<OwnedRepr<f64>, Ix1>]) -> Vec<f64> {
    let mut output = Vec::<f64>::new();
    for weight in weights {
        output.push(input.dot(weight));
    }
    return output
}