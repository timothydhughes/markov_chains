use crate::stochastic_error::StochasticError;
use approx::abs_diff_ne;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimName, MatrixN, RealField};
use std::fmt;

// A right stochastic matrix.
// NewType f64 only positive and 0 < x < 1?
/// A right stochastic matrix. Generic in dimensions and numeric type, though numeric must be a float.
#[derive(Debug, Clone, Hash)]
pub struct StochasticMatrix<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: Dim + DimName,
    DefaultAllocator: Allocator<N, D, D>,
{
    // always 2 dimensional
    internal: MatrixN<N, D>,
    // number of rows/columns
    n: usize,
}

// Placeholder, can probably be done better
impl<N, D> fmt::Display for StochasticMatrix<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: Dim + DimName,
    DefaultAllocator: Allocator<N, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.internal)
    }
}

impl<N, D> StochasticMatrix<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: Dim + DimName,
    DefaultAllocator: Allocator<N, D, D>,
{
    // Consumes given matrix
    // TODO: Take any floating point type, not just f64
    /// Consumes given matrix and attempts to create a StochasticMatrix.
    pub fn new(matrix: MatrixN<N, D>) -> Result<StochasticMatrix<N, D>, StochasticError>
    where
        <D as nalgebra::base::dimension::DimName>::Value: std::ops::Mul,
    {
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
        Ok(StochasticMatrix {
            internal: matrix,
            n,
        })
    }
    // TODO: Remove clones
    /// Returns cloned internal matrix.
    #[inline]
    pub fn internal(&self) -> MatrixN<N, D> {
        self.internal.clone()
    }

    /// Returns size of row of matrix.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }
}
