#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces
)]
/*!
# `m_c` #

A library for creating and using Markov chains and realizations of other stochastic processes.

## Using `m_c` ##

The most important type is the (private) `MarkovChain` struct. Using the struct is
done directly through the two exported types `DynamicMarkovChain` and
`StaticMarkovChain` along with the methods defined on the `Chain` trait.

```.rust
use m_c::{Chain, StaticMarkovChain, StochasticMatrix, StationaryOptions};
use m_c::na::{Matrix2, Vector2};
use std::error::Error;

// Finds the stationary distribution of a given matrix
fn main() -> Result<(), Box<dyn Error>> {
    // Note column-major order
    let vec = vec![0.2, 0.3, 0.8, 0.7];
    let matrix = StochasticMatrix::new(Matrix2::from_vec(vec))?;
    let chain = StaticMarkovChain::new(matrix)?;
    let stationary_distribution = chain
        .stationary_distribution(StationaryOptions::Default)?;
    // Checks floats for equality, not the best idea
    // In this case (on my machine) it works, though
    let ans_vec = vec![0.2727272727272727, 0.7272727272727272];
    let answer = Vector2::from_vec(ans_vec);
    assert_eq!(answer, stationary_distribution);
    Ok(())
}
```

*/
// TODO: Rework comparisons
// Tests fail due to sums not being close enough to 1.
mod categorical;
mod markov_chain;
mod stochastic_error;
mod stochastic_matrix;

pub use markov_chain::Chain;
pub use markov_chain::{DynamicMarkovChain, StaticMarkovChain, StationaryOptions};
pub use nalgebra as na;
pub use rand as rn;
pub use stochastic_error::StochasticError;
pub use stochastic_matrix::{DynamicStochasticMatrix, StochasticMatrix};
pub use typenum as tn;
