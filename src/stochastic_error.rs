use thiserror::Error;

/// The error type for the crate.
///
/// Errors can occur during many operations, however, most can occur during instantiation of types.
#[derive(Debug, Error, Clone, Copy)]
pub enum StochasticError {
    /// Occurs when creating a stochastic matrix. Matrix type must be square.
    #[error("Given matrix is not square")]
    NotSquare,
    /// Occurs when creating a stochastic matrix. Matrix rows must sum to 1.
    /// Associated value is the offending row.
    #[error("Given matrix has a row that does not sum to 1: row {0}")]
    RowDoesNotSum1(usize),
    /// Occurs when creating a stochastic matrix. The matrix cannot contain negative values.
    #[error("Given matrix has a negative value")]
    NegativeValue,
    /// Occurs when creating a stochastic matrix. A matrix must be of size 2 or greater.
    /// Associated value is the offending length.
    #[error("Given length is too small: {0} must be greater than or equal to 2")]
    GivenLengthTooSmall(usize),
    /// Occurs when attempting to solve for the stationary distribution. The chain must be irreducible.
    #[error("Given matrix has more than one communication class.")]
    NotIrreducible,
    /// Occurs when solver method is unable to converge for stationary distribution.
    #[error("Unable to determine stationary distribution. Given matrix does not have limiting distribution equal to stationary distribution")]
    UnableToSolve,
    /// Occurs when passing a negative parameter or value greater than 1 to the **StationaryOptions** enum.
    #[error("Given threshold is either negative or greater than 1")]
    InvalidThreshold(f64),
    /// Occurs when attempting to sample from a vector which is not the same length as relevant matrix.
    #[error("Given vector does not have equal length to matrix")]
    LengthNotEqual,
    #[error("Unable to convert given value")]
    /// Occurs when attempting to downcast a given type to an ```f64``` and a candidate is not found.
    /// See ```try_convert()``` in _nalgebra_ for more info.
    ConversionError,
    /// Occurs when passing an invalid index for the underlying matrix.
    #[error("Given index is out of bounds: {0}")]
    IndexOutOfBounds(usize),
}
