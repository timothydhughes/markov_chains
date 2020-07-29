use super::{
    common::{
        aperiodic, can_transition_to as ctt, communicates_with as cw, create_graph, period as per,
        IntoIter,
    },
    Categorical, Chain, DynamicMarkovChain, DynamicStochasticMatrix, MarkovChain,
    StationaryOptions, StochasticError,
};
use approx::abs_diff_ne;
use nalgebra::Dynamic;
use nalgebra::{try_convert, DVector, RealField};
use petgraph::dot::Dot;
use petgraph::graph::NodeIndex;
use rand::thread_rng;
use rand::Rng;
use rand_distr::Distribution;
use std::fmt;

impl<N> fmt::Debug for DynamicMarkovChain<N>
where
    N: RealField + std::convert::From<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MarkovChain")
            .field("Matrix", &self.matrix)
            .field("States", &self.states)
            .field("Categorical Distribution", &self.categorical)
            .field("Graph (dot format)", &Dot::with_config(&self.graph, &[]))
            .finish()
    }
}

impl<N> Chain<N> for DynamicMarkovChain<N>
where
    N: RealField + std::convert::From<f64>,
{
    type StochasticMatrix = DynamicStochasticMatrix<N>;
    type StationaryDistribution = DVector<N>;
    type ChainIterator = IntoIter<DynamicMarkovChain<N>, N, Dynamic>;

    new_markov_chain!();
    sample_state!();
    sample_state_distribution!();
    small_helpers!();
    stationary_distribution!();
}

// TODO: Figure out way to deduplicate
// Probably impossible for now on stable Rust
impl<N> DynamicMarkovChain<N>
where
    N: RealField + std::convert::From<f64>,
{
    fn stationary_distribution_power(
        &self,
        tolerance: f64,
        iterations: usize,
    ) -> Result<DVector<N>, StochasticError> {
        if tolerance < 0.0 || tolerance > 1.0 {
            return Err(StochasticError::InvalidThreshold(tolerance));
        }
        if !self.is_irreducible() {
            return Err(StochasticError::NotIrreducible);
        }
        let matrix = self.matrix.internal().transpose();
        let mut res: DVector<N> = DVector::zeros(matrix.nrows());
        res[0] = 1.0_f64.into();
        let mut res_test: DVector<N> = DVector::zeros(matrix.nrows());
        let mut test = &res - res_test;
        let mut testval = test.norm();
        let mut iteration = 0;
        while iteration < iterations {
            if testval < tolerance.into() {
                break;
            }
            res_test = &matrix * &res;
            res = &matrix * &res_test;
            test = &res - &res_test;
            testval = test.norm();
            iteration += 1;
        }
        if matrix * &res != res {
            return Err(StochasticError::UnableToSolve);
        }
        Ok(res)
    }
    sample_distribution_times!(DVector<N>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    #[test]
    fn distribution() {
        let matrix = DynamicStochasticMatrix::from_vec(
            vec![0.33, 0.0, 1.0, 0.33, 0.5, 0.0, 0.34, 0.5, 0.0],
            3,
        )
        .unwrap();
        let chain = DynamicMarkovChain::new(matrix).unwrap();
        let initial = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let result = chain.sample_distribution_times(initial, 1);
        let test = DVector::from_vec(vec![0.33, 0.33, 0.34]);
        assert_eq!(result.unwrap(), test);
    }
    #[test]
    fn distribution_size_not_equal() {
        let matrix = DynamicStochasticMatrix::from_vec(
            vec![0.33, 0.0, 1.0, 0.33, 0.5, 0.0, 0.34, 0.5, 0.0],
            3,
        )
        .unwrap();
        let chain = DynamicMarkovChain::new(matrix).unwrap();
        let initial = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let result = chain.sample_distribution_times(initial, 1);
        assert!(result.is_err());
    }
}
