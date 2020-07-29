use super::{
    common::{
        aperiodic, can_transition_to as ctt, communicates_with as cw, create_graph, period as per,
        IntoIter,
    },
    Categorical, Chain, MarkovChain, StaticMarkovChain, StationaryOptions, StochasticError,
    StochasticMatrix,
};
use approx::abs_diff_ne;
use nalgebra::allocator::Allocator;
use nalgebra::{try_convert, DefaultAllocator, DimName, RealField, VectorN};
use petgraph::dot::Dot;
use petgraph::graph::NodeIndex;
use rand::thread_rng;
use rand::Rng;
use rand_distr::Distribution;
use std::fmt;

impl<N, D> fmt::Debug for StaticMarkovChain<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: DimName,
    DefaultAllocator: Allocator<N, D, D>,
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

impl<N, D> Chain<N> for StaticMarkovChain<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: DimName,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    type StochasticMatrix = StochasticMatrix<N, D>;
    type StationaryDistribution = VectorN<N, D>;
    type ChainIterator = IntoIter<StaticMarkovChain<N, D>, N, D>;
    new_markov_chain!();
    sample_state!();
    sample_state_distribution!();
    small_helpers!();
    stationary_distribution!();
}

impl<N, D> StaticMarkovChain<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: DimName,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    // https://en.wikipedia.org/wiki/Power_iteration
    fn stationary_distribution_power(
        &self,
        tolerance: f64,
        iterations: usize,
    ) -> Result<VectorN<N, D>, StochasticError> {
        if tolerance < 0.0 || tolerance > 1.0 {
            return Err(StochasticError::InvalidThreshold(tolerance));
        }
        if !self.is_irreducible() {
            return Err(StochasticError::NotIrreducible);
        }
        // Left eigenvector, hence transpose
        let matrix = self.matrix.internal().transpose();
        let mut res: VectorN<N, D> = VectorN::zeros();
        // Start with 1. as the eigenvector should sum to 1. anyways.
        res[0] = 1.0_f64.into();
        let mut res_test: VectorN<N, D> = VectorN::zeros();
        let mut test = &res - res_test;
        let mut testval = test.norm();
        let mut iteration = 0;
        while iteration < iterations {
            // Check for early convergence
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

    // distribution must sum to 1
    sample_distribution_times!(VectorN<N, D>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{zero, Matrix3, VectorN, U3};
    #[test]
    fn distribution() {
        let matrix2 = StochasticMatrix::new(Matrix3::from_vec(vec![
            0.33, 0.0, 1.0, 0.33, 0.5, 0.0, 0.34, 0.5, 0.0,
        ]))
        .unwrap();
        let chain = StaticMarkovChain::new(matrix2).unwrap();
        // this is unfortunate, related to this issue:
        // https://github.com/rustsim/nalgebra/issues/317
        let mut initial: VectorN<f64, _> = zero();
        initial[0] = 1.0;
        let result = chain.sample_distribution_times(initial, 1);
        let mut test: VectorN<f64, U3> = zero();
        test[0] = 0.33;
        test[1] = 0.33;
        test[2] = 0.34;
        assert_eq!(result.unwrap(), test);
    }
}
