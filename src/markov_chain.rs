use crate::categorical::Categorical;
use crate::stochastic_error::StochasticError;
use crate::stochastic_matrix::{DynamicStochasticMatrix, StochasticMatrix};
use nalgebra::RealField;
use petgraph::graph::Graph;
use rand::Rng;
use sealed::Sealed;

/// Convenience type for a markov chain backed by a statically sized stochastic matrix
pub type StaticMarkovChain<N, D> = MarkovChain<StochasticMatrix<N, D>>;

/// Convenience type for a markov chain backed by a dynamically sized stochastic matrix
pub type DynamicMarkovChain<N> = MarkovChain<DynamicStochasticMatrix<N>>;

/// An enum containing options used for solving for the stationary distribution of a Markov chain
#[derive(Debug, Clone, Copy)]
pub enum StationaryOptions {
    /// Uses the power method with 0.000_000_000_000_000_01 tolerance and 100_000 iterations.
    Default,
    /// Use the power method with (tolerance, iterations). Repeatedly multiplies the vector and matrix until convergence (exponentiation).
    Power(f64, usize),
}

#[macro_use]
mod common;
mod dynamic_markov_chain;
mod sealed;
mod static_markov_chain;
// Discrete
// finite
// 0-indexed
// Perhaps turn T into trait bound
// Make struct that wraps both types?
#[derive(Clone)]
pub struct MarkovChain<T> {
    matrix: T,
    // size
    states: usize,
    categorical: Vec<Categorical>,
    graph: Graph<usize, f64>,
}

// TODO Consider changing type definitions to newtype structs
// TODO -- https://stackoverflow.com/questions/56728840/how-can-i-implement-traits-for-type-aliases
/// The trait implemented by Markov chains.
pub trait Chain<N>: Sealed
where
    N: RealField,
{
    // Change to matrix type, not mark
    /// The type of Stochastic Matrix backing this specific Markov chain.
    type StochasticMatrix;
    /// The type of Vector returned by the ```stationary_distribution()``` method. Relates to matrix.
    type StationaryDistribution;
    /// The type of iterator returned by ```into_iter()``` method.
    type ChainIterator;

    /// Constructs a new Markov Chain instance.
    fn new(matrix: Self::StochasticMatrix) -> Result<Self, StochasticError>
    where
        Self: std::marker::Sized;
    /// Samples categorical distribution for given state. Returns next state.
    fn sample_state(&self, state: usize) -> Result<usize, StochasticError>;
    /// Samples given categorical distribution, then returns next state given that first result.
    fn sample_state_distribution<R: Rng + ?Sized>(
        &self,
        initial_distribution: &[f64],
        rng: &mut R,
    ) -> Result<usize, StochasticError>;
    /// Returns whether state ```from``` can transition to state ```to```.
    fn can_transition_to(&self, from: usize, to: usize) -> bool;
    /// Returns whether state ```from``` communicates with state ```to```.
    /// In other words, whether they are in the same communication class,
    /// or they both are in the same strongly connected component.
    fn communicates_with(&self, from: usize, to: usize) -> bool;
    /// Returns number of communication classes of array.
    fn communication_classes(&self) -> usize;
    /// Returns whether the given chain is irreducible or not, i.e. 1 communication class.
    fn is_irreducible(&self) -> bool {
        self.communication_classes() == 1
    }
    /// Returns the period of the Markov chain. Chain must be irreducible.
    /// Implementation based off [these notes](http://cecas.clemson.edu/~shierd/Shier/markov.pdf).
    fn period(&self) -> Result<u32, StochasticError>;
    /// Returns whether the Markov chain is aperiodic i.e. period of 1.
    fn is_aperiodic(&self) -> Result<bool, StochasticError>;
    /// Attempts to compute the stationary (limiting) distribution of the given Markov Chain.
    ///
    /// See [StationaryOptions](enum.StationaryOptions.html) for more information.
    fn stationary_distribution(
        &self,
        options: StationaryOptions,
    ) -> Result<Self::StationaryDistribution, StochasticError>;
    /// Consumes chain and returns an iterator.
    fn into_iter(self, current_state: usize) -> Self::ChainIterator;
}

#[cfg(test)]
mod tests {
    use super::Chain;
    use super::StationaryOptions;
    use super::{DynamicMarkovChain, StaticMarkovChain};
    use super::{DynamicStochasticMatrix, StochasticMatrix};
    use nalgebra::Vector3;
    use nalgebra::{DVector, Matrix2, Matrix3, Matrix4, Matrix5};
    #[test]
    fn chain_sample() {
        let matrix = StochasticMatrix::new(Matrix2::from_element(0.5)).unwrap();
        let matrix2 = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.1, 1.0, 0.2, 0.2, 0.0, 0.4, 0.7, 0.0, 0.4,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix);
        assert!(chain.is_ok());
        let chain2 = StaticMarkovChain::new(matrix2).unwrap();
        assert_eq!(chain2.sample_state(1).unwrap(), 0);
    }
    #[test]
    fn chain_sample_distribution() {
        let matrix = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.1, 1.0, 0.2, 0.2, 0.0, 0.4, 0.7, 0.0, 0.4,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        let mut rng = rand::thread_rng();
        let distribution = vec![0.0, 1.0, 0.0];
        assert_eq!(
            chain
                .sample_state_distribution(&distribution, &mut rng)
                .unwrap(),
            0
        );
    }

    #[test]
    fn chain_sample_distribution_err() {
        let matrix = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.1, 1.0, 0.2, 0.2, 0.0, 0.4, 0.7, 0.0, 0.4,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        let mut rng = rand::thread_rng();
        let distribution = vec![0.0];
        assert!(chain
            .sample_state_distribution(&distribution, &mut rng)
            .is_err());
    }
    #[test]
    fn stationary() {
        let matrix = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.5, 0.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        let answer: Vector3<_> = Vector3::from_vec(vec![0.25, 0.5, 0.25]);
        let result: Vector3<_> = chain
            .stationary_distribution(StationaryOptions::Default)
            .unwrap();
        assert_eq!(result, answer)
    }

    #[test]
    fn dynamic_stationary() {
        let matrix = DynamicStochasticMatrix::from_vec(
            vec![0.5, 0.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5],
            3,
        )
        .unwrap();
        let chain = DynamicMarkovChain::new(matrix).unwrap();
        let answer = DVector::from_vec(vec![0.25, 0.5, 0.25]);
        let result = chain
            .stationary_distribution(StationaryOptions::Default)
            .unwrap();
        assert_eq!(result, answer)
    }

    // how to handle limiting probabilities not equal to stationary?
    #[test]
    fn stationary2() {
        let matrix = StochasticMatrix::new(Matrix2::from_vec(vec![0.0, 1.0, 1.0, 0.0])).unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        assert!(chain
            .stationary_distribution(StationaryOptions::Power(0.10, 100))
            .is_err())
    }

    // #[test]
    // fn stationary_linear() {
    //     let matrix =
    //         StochasticMatrix::new(arr2(&[[0.7, 0.2, 0.1], [0.4, 0.6, 0.0], [0.0, 1.0, 0.0]]))
    //             .unwrap();
    //     let chain = MarkovChain::new(matrix);
    //     chain.stationary_distribution_linear();
    // }

    #[test]
    fn communication() {
        let matrix = StochasticMatrix::new(Matrix2::from_vec(vec![0.0, 1.0, 1.0, 0.0])).unwrap();
        let matrix2 = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.6, 0.8, 0.0, 0.4, 0.1, 0.0, 0.0, 0.1, 1.0,
        ]))
        .unwrap();
        let matrix3 = StochasticMatrix::new(Matrix5::from_vec(vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        let chain2 = StaticMarkovChain::new(matrix2).unwrap();
        let chain3 = StaticMarkovChain::new(matrix3).unwrap();
        assert_eq!(chain.communication_classes(), 1);
        assert_eq!(chain2.communication_classes(), 2);
        assert_eq!(chain3.communication_classes(), 5);
    }

    #[test]
    fn is_irreducible() {
        let matrix = StochasticMatrix::new(Matrix2::from_vec(vec![0.0, 1.0, 1.0, 0.0])).unwrap();
        let matrix2 = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.6, 0.8, 0.0, 0.4, 0.1, 0.0, 0.0, 0.1, 1.0,
        ]))
        .unwrap();
        let matrix3 = StochasticMatrix::new(Matrix5::from_vec(vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        let chain2 = StaticMarkovChain::new(matrix2).unwrap();
        let chain3 = StaticMarkovChain::new(matrix3).unwrap();
        assert!(chain.is_irreducible());
        assert!(!chain2.is_irreducible());
        assert!(!chain3.is_irreducible());
    }

    #[test]
    fn period_reducible() {
        let matrix = StochasticMatrix::new(Matrix2::from_vec(vec![1.0, 0.0, 0.0, 1.0])).unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        assert!(chain.period().is_err());
    }

    #[test]
    fn period_irreducible() {
        let matrix = StochasticMatrix::new(Matrix2::from_vec(vec![0.0, 1.0, 1.0, 0.0])).unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        assert_eq!(chain.period().unwrap(), 2);
    }

    #[test]
    fn period_irreducible_two() {
        let matrix = StochasticMatrix::new(Matrix4::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix).unwrap();
        assert_eq!(chain.period().unwrap(), 2);
    }
}
