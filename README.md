
# A Markov Chain Library

A library for creating and using Markov chains and realizations of other stochastic processes.

## Installation

As of right now, this crate isn't hosted on [crates.io](https://crates.io/). I personally don't feel as though its ready to publish (I'm not an expert in this subject and the name isn't good). If you wish to use it, clone this repo and use [cargo's ability to pull local crates in](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html) to experiment with it.

## Usage

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

View the tests and the library's documentation to see other examples of how to use the library properly. If you know the size of your matrix, use the static types, otherwise, use dynamic. The crate re-exports three libraries at the root, and the crate as it is would not be possible without them. Its best to use the re-exported crates so the versions stays compatible. As of right now  [specialization](https://github.com/rust-lang/rust/issues/31844), [const generics](https://github.com/rust-lang/rust/issues/44580), and [generic associated types](https://github.com/rust-lang/rust/issues/44265) would make this library a lot more ergonomic. If you do use this library and discover any issues, please open an issue on GitHub or contact me.

## Contributing

Pull requests are welcome. Currently, various TODOs are in the code that I haven't gotten around to. In the future, I hope to add more than just Markov chains, e.g. n-dimensional Brownian motion, random walks, Poisson processes.

## License

[MIT](https://choosealicense.com/licenses/mit/)
