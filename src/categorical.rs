use crate::stochastic_error::StochasticError;
use approx::abs_diff_ne;
use rand::distributions::Standard;
use rand::prelude::*;
use rand_distr::Distribution;

// TODO: Test for bias. Long run results should approach initial distribution.
/// A categorical distribution.
#[derive(Debug, Clone)]
pub(crate) struct Categorical {
    /// Size of sample space
    n: usize,
    /// Vector of probabilities corresponding to results
    probs: Vec<f64>,
    /// Vector of Cumulative probabilities corresponding to results. For inverse sampling.
    cumulative: Vec<f64>,
}

impl Categorical {
    /// Returns a categorical distribution with probabiltities corresponding to given vector.
    /// # Arguments
    ///
    /// * `probs_slice` - A slice that holds the probabilities corresponding to each discrete event. Must sum to 1 and contain no negatives.
    ///
    pub(crate) fn new(probs_slice: &[f64]) -> Result<Self, StochasticError> {
        let sum: f64 = probs_slice.iter().sum();
        for i in probs_slice {
            if i < &0.0 {
                return Err(StochasticError::NegativeValue);
            }
        }
        if abs_diff_ne!(sum, 1.0) {
            return Err(StochasticError::RowDoesNotSum1(0));
        }
        let probs = probs_slice.to_vec();
        let n = probs.len();
        let mut cumulative = Vec::with_capacity(n);
        for i in 0..n {
            if i == 0 {
                cumulative.push(probs[0]);
            } else if i == n {
                cumulative.push(1.0);
            } else {
                cumulative.push(cumulative[i - 1] + probs[i])
            }
        }
        Ok(Categorical {
            n,
            probs,
            cumulative,
        })
    }
}

impl Distribution<usize> for Categorical {
    // TODO: Implement binary search
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let sample: f64 = rng.sample(Standard);
        if sample < self.cumulative[0] {
            return 0;
        }
        let mut i = 1;
        while i < self.n {
            if sample > self.cumulative[i - 1] && sample < self.cumulative[i] {
                return i;
            }
            i += 1;
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::Categorical;

    #[test]
    fn categorical_err() {
        let vec = vec![0.1, 0.3, 0.4, 0.4];
        let categorical = Categorical::new(&vec);
        assert!(categorical.is_err());
    }

    #[test]
    fn categorical_err2() {
        let vec = vec![0.1, 0.3, -0.4, 0.4];
        let categorical = Categorical::new(&vec);
        assert!(categorical.is_err());
    }

    #[test]
    fn cumulative() {
        let vec = vec![0.1, 0.3, 0.4, 0.2];

        let cumulative = vec![0.1, 0.4, 0.8, 1.0];
        let categorical = Categorical::new(&vec).unwrap();
        assert_eq!(categorical.cumulative, cumulative);
    }
}
