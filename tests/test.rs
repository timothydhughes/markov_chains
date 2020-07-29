extern crate m_c;

use m_c::na::{Matrix2, Vector2};
use m_c::rn::prelude::*;
use m_c::{Chain, DynamicMarkovChain, StaticMarkovChain, StationaryOptions};
use m_c::{DynamicStochasticMatrix, StochasticMatrix};

#[test]
fn create_dynamic_chain() {
    let mut rng = StdRng::from_entropy();
    let matrix: DynamicStochasticMatrix<f64> =
        DynamicStochasticMatrix::new_random(3, &mut rng).unwrap();
    let chain = DynamicMarkovChain::new(matrix).unwrap();
    chain.sample_state(0).unwrap();
}

#[test]
fn create_static_chain() {
    let vec = vec![0.2, 0.3, 0.8, 0.7];
    let matrix = StochasticMatrix::new(Matrix2::from_vec(vec)).unwrap();
    let chain = StaticMarkovChain::new(matrix).unwrap();
    let stationary_distribution = chain
        .stationary_distribution(StationaryOptions::Default)
        .unwrap();
    let ans_vec = vec![0.2727272727272727, 0.7272727272727272];
    let answer = Vector2::from_vec(ans_vec);
    assert_eq!(answer, stationary_distribution);
}

#[test]
fn create_static_chain_iterator() {
    let vec = vec![0.0, 1.0, 1.0, 0.0];
    let matrix = StochasticMatrix::new(Matrix2::from_vec(vec)).unwrap();
    let chain = StaticMarkovChain::new(matrix).unwrap();
    let mut iterator = Vec::new();
    for state in chain.into_iter(0).take(5) {
        iterator.push(state);
    }
    assert_eq!(iterator, vec![1, 0, 1, 0, 1])
}

#[test]
fn create_dynamic_chain_iterator() {
    let vec = vec![0.0, 1.0, 1.0, 0.0];
    let matrix: DynamicStochasticMatrix<f64> = DynamicStochasticMatrix::from_vec(vec, 2).unwrap();
    let chain = DynamicMarkovChain::new(matrix).unwrap();
    let mut iterator = Vec::new();
    for state in chain.into_iter(0).take(5) {
        iterator.push(state);
    }
    assert_eq!(iterator, vec![1, 0, 1, 0, 1])
}
