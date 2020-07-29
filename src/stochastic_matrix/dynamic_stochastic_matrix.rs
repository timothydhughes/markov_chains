use crate::stochastic_error::StochasticError;
use approx::abs_diff_ne;
use nalgebra::{convert, DMatrix, RealField};
use rand::prelude::*;
use std::fmt;

#[derive(Debug, Clone, Hash)]
/// A right stochastic matrix, with dynamically sized array backing.
///
/// Generic in numeric type.
pub struct DynamicStochasticMatrix<N>
where
    N: RealField + std::convert::From<f64>,
{
    internal: DMatrix<N>,
    n: usize,
}

// Placeholder, can probably be done better
impl<N> fmt::Display for DynamicStochasticMatrix<N>
where
    N: RealField + std::convert::From<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.internal)
    }
}

impl<N> DynamicStochasticMatrix<N>
where
    N: RealField + std::convert::From<f64>,
{
    /// Consumes given matrix and attempts to create an instance of a Stochastic Matrix
    pub fn new(matrix: DMatrix<N>) -> Result<DynamicStochasticMatrix<N>, StochasticError> {
        if !matrix.is_square() {
            return Err(StochasticError::NotSquare);
        }
        let n = matrix.shape().0;
        for (index, row) in matrix.row_iter().enumerate() {
            if abs_diff_ne!(row.sum(), 1.0_f64.into()) {
                return Err(StochasticError::RowDoesNotSum1(index));
            }
            for e in row.iter() {
                if e < &0.0_f64.into() || e > &1.0_f64.into() {
                    return Err(StochasticError::NegativeValue);
                }
            }
        }
        Ok(DynamicStochasticMatrix {
            internal: matrix,
            n,
        })
    }

    // TODO: Find way to allow rust to infer type, Remove type from
    // test tests::stochastic_rand_matrix_row
    // to see error thrown as example.
    /// Creates new dynamically sized stochastic matrix from given rng and size
    /// Vectors are uniformly distributed according to a dirichlet distribution.
    pub fn new_random<R: Rng + ?Sized>(
        n: usize,
        rng: &mut R,
    ) -> Result<DynamicStochasticMatrix<N>, StochasticError> {
        if n < 2 {
            return Err(StochasticError::GivenLengthTooSmall(n));
        }
        let mut arr_vec = Vec::with_capacity(n * n);
        for _ in 0..n {
            let mut vec = DynamicStochasticMatrix::stochastic_vector(n, rng);
            arr_vec.append(&mut vec);
        }
        // from_vec fills by column order
        // hence, take the transpose
        let internal = DMatrix::from_vec(n, n, arr_vec).transpose();
        Ok(DynamicStochasticMatrix { internal, n })
    }

    /// Creates new dynamically sized stochastic matrix from a given vector.
    ///
    /// Given vector must have a perfect square length, and fulfill all other requirements.
    /// Given vector must also be in row major order.
    /// Parameter n is length of a row.
    pub fn from_vec(vec: Vec<N>, n: usize) -> Result<DynamicStochasticMatrix<N>, StochasticError> {
        if n * n != vec.len() {
            return Err(StochasticError::NotSquare);
        }
        // Check rows sum to 1
        // Check for negative values
        // Check for numbers greater than 1
        for item in vec.iter() {
            if item < &0.0_f64.into() || item > &1.0f64.into() {
                return Err(StochasticError::NegativeValue);
            }
        }

        for i in 0..n {
            let mut sum: N = 0.0.into();
            for k in 0..n {
                sum += vec[i + (n * k)];
            }
            if abs_diff_ne!(sum, 1.0_f64.into()) {
                return Err(StochasticError::RowDoesNotSum1(i));
            }
        }
        let matrix = DMatrix::from_vec(n, n, vec);
        Ok(DynamicStochasticMatrix {
            internal: matrix,
            n,
        })
    }

    // use dirichlet distribution
    // error if n < 2
    pub(crate) fn stochastic_vector<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<N> {
        // will not panic as alpha is valid
        let alpha: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let dirich = rand_distr::Dirichlet::new(alpha).unwrap();
        let inter = dirich.sample(rng);
        inter.into_iter().map(convert).collect()
    }
    /// Returns cloned internal matrix.
    #[inline]
    pub fn internal(&self) -> DMatrix<N> {
        self.internal.clone()
    }

    /// Returns size of row of matrix.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }
}
