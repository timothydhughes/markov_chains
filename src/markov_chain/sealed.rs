use super::{DynamicMarkovChain, StaticMarkovChain};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimName, RealField};
pub trait Sealed {}

// Keep the trait in crate only.
impl<N> Sealed for DynamicMarkovChain<N> where N: RealField + std::convert::From<f64> {}
impl<N, D> Sealed for StaticMarkovChain<N, D>
where
    N: RealField + std::convert::From<f64>,
    D: Dim + DimName,
    DefaultAllocator: Allocator<N, D, D>,
{
}
