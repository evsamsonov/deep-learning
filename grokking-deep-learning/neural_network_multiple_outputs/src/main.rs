extern crate ndarray;
extern crate blas_src;

use ndarray::{Array, ArrayBase, OwnedRepr, Ix1};

// see https://github.com/rust-ndarray/ndarray
// see https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html
// need install openblas lib

fn main() {
    let win_shares = [0.65, 0.8, 0.8, 0.9];

    let weights = Array::from_vec(vec![0.3, 0.2, 0.9]);
    let input = win_shares[0];
    let prediction = neural_network(input, weights);
    println!("{}", prediction)
}

fn neural_network(input: f64, weights: ArrayBase<OwnedRepr<f64>, Ix1>) -> ArrayBase<OwnedRepr<f64>, Ix1> {
    return weights * input
}