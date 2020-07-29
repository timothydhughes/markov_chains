mod dynamic_stochastic_matrix;
mod static_stochastic_matrix;

pub use dynamic_stochastic_matrix::DynamicStochasticMatrix;
pub use static_stochastic_matrix::StochasticMatrix;

#[cfg(test)]
mod tests {
    use super::dynamic_stochastic_matrix::*;
    use super::static_stochastic_matrix::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::{Matrix2, Matrix3, MatrixMN, U127};
    use rand::prelude::*;
    // Try something actually large
    type LargeMatrix = MatrixMN<f64, U127, U127>;
    // sum of row must be 1
    #[test]
    fn stochastic_matrix_row_sum() {
        let matrix = StochasticMatrix::new(Matrix2::from_element(0.5));
        let matrix2 = StochasticMatrix::new(Matrix2::from_element(0.2));
        let matrix3 = StochasticMatrix::new(Matrix2::from_element(-0.2));
        assert!(matrix.is_ok());
        assert!(matrix2.is_err());
        assert!(matrix3.is_err());
    }
    // Type system does not allow
    // Test dynamic instead
    // #[test]
    // fn stochastic_matrix_square() {
    //     let matrix2 = StochasticMatrix::new(Matrix2x3::from_element(0.33333333));
    //     assert!(matrix2.is_err());
    // }

    #[test]
    fn stochastic_matrix_large() {
        let matrix: LargeMatrix = LargeMatrix::identity();
        let smatrix = StochasticMatrix::new(matrix);
        assert!(smatrix.is_ok())
    }

    #[test]
    fn stochastic_matrix_negative() {
        let matrix =
            StochasticMatrix::new(Matrix3::new(0.0, 0.2, 0.8, 0.5, 0.5, 0.0, -0.2, 0.6, 0.6));
        assert!(matrix.is_err());
    }
    // fails approx 1 in ~100 due to floating point issues?
    // due to properties of the Dirchlet distribution, we know that the vector is "correct"
    #[test]
    fn stochastic_vector_sum() {
        let mut rng = StdRng::from_entropy();
        let vector = DynamicStochasticMatrix::stochastic_vector(3, &mut rng);
        let sum: f64 = vector.iter().sum();
        assert_abs_diff_eq!(sum, 1.)
    }

    #[test]
    fn stochastic_vector_len() {
        let mut rng = StdRng::from_entropy();
        let n = rng.gen_range(2, 100);
        let vector: Vec<f64> = DynamicStochasticMatrix::stochastic_vector(n, &mut rng);
        assert!(vector.len() == n)
    }
    #[test]
    fn dynamic_stochastic_rand_matrix_size() {
        let mut rng = StdRng::from_entropy();
        let n = rng.gen_range(2, 100);
        let matrix: DynamicStochasticMatrix<f64> =
            DynamicStochasticMatrix::new_random(n, &mut rng).unwrap();
        let shape = matrix.internal().shape();
        assert_eq!(shape, (n, n))
    }

    // Test for sum of rows is equal to 1
    #[test]
    fn dynamic_stochastic_rand_matrix_row() {
        let mut rng = StdRng::from_entropy();
        let n = rng.gen_range(2, 100);
        let matrix: DynamicStochasticMatrix<f64> =
            DynamicStochasticMatrix::new_random(n, &mut rng).unwrap();
        for row in matrix.internal().row_iter() {
            assert_relative_eq!(row.sum(), 1., epsilon = std::f64::EPSILON * 4.)
        }
    }

    #[test]
    fn dynamic_stochastic_matrix_from_vec() {
        let vec = vec![0.1, 0.7, 0.9, 0.3];
        let vec2 = vec![0.2, 0.3, 0.4];
        let matrix = DynamicStochasticMatrix::from_vec(vec, 2);
        let matrix2 = DynamicStochasticMatrix::from_vec(vec2.clone(), 2);
        let matrix3 = DynamicStochasticMatrix::from_vec(vec2, 1);
        assert!(matrix.is_ok());
        assert!(matrix2.is_err());
        assert!(matrix3.is_err());
    }
}
