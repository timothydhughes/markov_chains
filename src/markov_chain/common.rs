use crate::markov_chain::Chain;
use crate::StochasticError;
use approx::abs_diff_eq;
use nalgebra::{allocator::Allocator, try_convert, DefaultAllocator, Dim, MatrixN, RealField};
use num::integer::gcd;
use petgraph::algo::has_path_connecting;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::Dfs;
use std::collections::{BTreeSet, VecDeque};
use std::marker::PhantomData;

pub(crate) fn create_graph<N, D>(
    matrix: &MatrixN<N, D>,
    n: usize,
) -> Result<Graph<usize, f64>, StochasticError>
where
    N: RealField + std::convert::From<f64>,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    let mut graph: Graph<usize, f64, petgraph::Directed, u32> = Graph::with_capacity(n, n);
    for i in 0..n {
        graph.add_node(i);
    }
    for (state, row) in matrix.row_iter().enumerate() {
        for i in 0..n {
            // If the weight is 0, skip the edge
            if abs_diff_eq!(
                try_convert(row[i]).ok_or(StochasticError::ConversionError)?,
                0.0
            ) {
                continue;
            }
            let from = NodeIndex::new(state);
            let to = NodeIndex::new(i);
            graph.add_edge(
                from,
                to,
                try_convert(row[i]).ok_or(StochasticError::ConversionError)?,
            );
        }
    }
    Ok(graph)
}

#[inline]
pub(crate) fn can_transition_to<N, D>(matrix: &MatrixN<N, D>, from: usize, to: usize) -> bool
where
    N: RealField + std::convert::From<f64>,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    matrix.index((from, to)) > &0.0.into()
}

#[inline]
pub(crate) fn communicates_with(graph: &Graph<usize, f64>, from: u32, to: u32) -> bool {
    has_path_connecting(graph, from.into(), to.into(), None)
        && has_path_connecting(graph, to.into(), from.into(), None)
}

struct Node {
    id: NodeIndex<u32>,
    level: u32,
}

impl Node {
    fn new(id: NodeIndex<u32>, level: u32) -> Node {
        Node { id, level }
    }

    // self should be lower on the tree i.e. higher level
    // doesn't really matter
    fn e_value(&self, other: &Node) -> u32 {
        if self.level > other.level {
            self.level - other.level + 1
        } else {
            other.level - self.level + 1
        }
    }
}

#[derive(Debug)]
struct StackNode {
    parent: NodeIndex<u32>,
    node: NodeIndex<u32>,
}

impl StackNode {
    fn new(parent: NodeIndex<u32>, node: NodeIndex<u32>) -> StackNode {
        StackNode { parent, node }
    }
}

// Implementation based off http://cecas.clemson.edu/~shierd/Shier/markov.pdf
// Probably a more efficient implementation possible. Using visit::depth_first_search from petgraph
pub(crate) fn period(graph: &Graph<usize, f64>, start: NodeIndex<u32>) -> u32 {
    // construct a bfs tree
    // we do it manually because we get access to more information
    // edge weight is false for down-edge, true otherwise
    let mut bfs: Graph<Node, bool> = Graph::new();
    let root_node = Node::new(start, 0);
    let root_bfs = bfs.add_node(root_node);
    let mut level: u32;
    let mut stack: VecDeque<StackNode> = VecDeque::new();
    for node in graph.neighbors(start) {
        stack.push_back(StackNode::new(root_bfs, node));
    }
    let mut seen: BTreeSet<NodeIndex<u32>> = BTreeSet::new();
    seen.insert(start);
    while !stack.is_empty() {
        // Cannot fail
        let current = stack.pop_front().unwrap();
        if !seen.contains(&current.node) {
            level = bfs[current.parent].level + 1;
            let current_bfs_node = Node::new(current.node, level);
            let bfs_node_ix = bfs.add_node(current_bfs_node);
            bfs.add_edge(current.parent, bfs_node_ix, false);
            for neighbor in graph.neighbors(current.node) {
                if seen.contains(&neighbor) {
                    let mut dfs_search = Dfs::new(&bfs, root_bfs);
                    // find index for corresponding value in tree
                    while let Some(nx) = dfs_search.next(&bfs) {
                        if bfs[nx].id == neighbor {
                            bfs.add_edge(bfs_node_ix, nx, true);
                            break;
                        }
                    }
                } else {
                    stack.push_back(StackNode::new(bfs_node_ix, neighbor));
                }
            }
            seen.insert(current.node);
        }
    }
    // after constructing tree, search for non-down edges
    let mut set_k = Vec::new();
    for edge in bfs.edge_indices() {
        let edge_weight = bfs.edge_weight(edge).unwrap();
        // when we find one, add computed edge value to vec k
        if *edge_weight {
            let endpoints = bfs.edge_endpoints(edge).unwrap();
            let node = &bfs[endpoints.0];
            let node1 = &bfs[endpoints.1];
            let k_edge = node.e_value(&node1);
            set_k.push(k_edge);
        }
    }
    // period is gcd of the set
    set_k.into_iter().fold(0, gcd)
}

#[inline]
pub(crate) fn aperiodic(graph: &Graph<usize, f64>) -> bool {
    period(graph, NodeIndex::from(0)) == 1
}

#[derive(Debug)]
pub struct IntoIter<C, N, D>
where
    C: Chain<N>,
    N: RealField + std::convert::From<f64>,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    chain: C,
    current_state: usize,
    _marker: PhantomData<N>,
    _marker2: PhantomData<D>,
}

impl<C, N, D> IntoIter<C, N, D>
where
    C: Chain<N>,
    N: RealField + std::convert::From<f64>,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    pub fn new(chain: C, current_state: usize) -> IntoIter<C, N, D> {
        IntoIter {
            chain,
            current_state,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }
}

impl<C, N, D> Iterator for IntoIter<C, N, D>
where
    C: Chain<N>,
    N: RealField + std::convert::From<f64>,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.chain.sample_state(self.current_state);
        if next.is_err() {
            return None;
        }
        self.current_state = next.unwrap();
        Some(self.current_state)
    }
}

// Macros are to cut down on duplicate code. Readability suffers a bit.
macro_rules! new_markov_chain {
    () => {
        fn new(
            matrix: Self::StochasticMatrix,
        ) -> Result<MarkovChain<Self::StochasticMatrix>, StochasticError> {
            let states = matrix.size();
            let mut categorical: Vec<Categorical> = Vec::with_capacity(states);
            let view = matrix.internal();
            for row in view.row_iter() {
                let vec_probs: Vec<f64> = row
                    .iter()
                    .map(|x| try_convert(*x).ok_or(StochasticError::ConversionError))
                    .collect::<Result<Vec<f64>, StochasticError>>()?;
                let c = Categorical::new(&vec_probs);
                categorical.push(c?);
            }
            let graph = create_graph(&view, states)?;
            Ok(MarkovChain {
                matrix,
                states,
                categorical,
                graph,
            })
        }
    };
}

macro_rules! sample_distribution_times {
    ($ret_type:ty) => {
        pub fn sample_distribution_times(
            &self,
            initial_dist: $ret_type,
            power: usize,
        ) -> Result<$ret_type, StochasticError> {
            for i in &initial_dist {
                if i < &0.0.into() {
                    return Err(StochasticError::NegativeValue);
                }
            }
            if abs_diff_ne!(initial_dist.sum(), 1.0.into()) {
                return Err(StochasticError::RowDoesNotSum1(0));
            }
            // matrix always a square
            let rows = self.matrix.internal().nrows();
            if rows != initial_dist.len() {
                return Err(StochasticError::LengthNotEqual);
            }
            let mut result = initial_dist;
            let matrix = self.matrix.internal().transpose();
            for _ in 0..power {
                result = &matrix * result;
            }
            Ok(result)
        }
    };
}

macro_rules! sample_state {
    () => {
        fn sample_state(&self, current_state: usize) -> Result<usize, StochasticError> {
            if current_state >= self.states {
                return Err(StochasticError::IndexOutOfBounds(current_state));
            }
            let mut rng = thread_rng();
            Ok(self.categorical[current_state].sample(&mut rng))
        }
    };
}

macro_rules! sample_state_distribution {
    () => {
        fn sample_state_distribution<R: Rng + ?Sized>(
            &self,
            initial_distribution: &[f64],
            rng: &mut R,
        ) -> Result<usize, StochasticError> {
            if initial_distribution.len() != self.states {
                return Err(StochasticError::LengthNotEqual);
            }
            let c = Categorical::new(initial_distribution)?;
            Ok(self.categorical[c.sample(rng)].sample(rng))
        }
    };
}

macro_rules! stationary_distribution {
    () => {
        fn stationary_distribution(
            &self,
            options: StationaryOptions,
        ) -> Result<Self::StationaryDistribution, StochasticError> {
            match options {
                StationaryOptions::Default => {
                    // Probably ridiculous defaults?
                    self.stationary_distribution_power(0.000_000_000_000_000_01, 100_000)
                }
                StationaryOptions::Power(tol, iterations) => {
                    self.stationary_distribution_power(tol, iterations)
                }
            }
        }
    };
}

macro_rules! small_helpers {
    () => {
        fn can_transition_to(&self, from: usize, to: usize) -> bool {
            ctt(&self.matrix.internal(), from, to)
        }
        fn communicates_with(&self, from: usize, to: usize) -> bool {
            cw(&self.graph, from as u32, to as u32)
        }

        fn period(&self) -> Result<u32, StochasticError> {
            if !self.is_irreducible() {
                return Err(StochasticError::NotIrreducible);
            }
            Ok(per(&self.graph, NodeIndex::from(0)))
        }

        fn is_aperiodic(&self) -> Result<bool, StochasticError> {
            if !self.is_irreducible() {
                return Err(StochasticError::NotIrreducible);
            }
            Ok(aperiodic(&self.graph))
        }

        fn communication_classes(&self) -> usize {
            let classes = petgraph::algo::tarjan_scc(&self.graph);
            classes.len()
        }
        fn into_iter(self, current_state: usize) -> Self::ChainIterator {
            IntoIter::new(self, current_state)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynamicStochasticMatrix;
    use nalgebra::Matrix3;
    #[test]
    fn graph() {
        let matrix = DynamicStochasticMatrix::from_vec(
            vec![0.5, 0.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5],
            3,
        )
        .unwrap();
        let statmatrix = Matrix3::from_vec(vec![0.5, 0.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5]);
        assert!(create_graph(&matrix.internal(), 3).is_ok());
        assert!(create_graph(&statmatrix, 3).is_ok());
    }

    #[test]
    fn transition() {
        let matrix = Matrix3::from_vec(vec![0.5, 0.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5]);
        assert!(!can_transition_to(&matrix, 0, 2));
        assert!(can_transition_to(&matrix, 0, 0));
        assert!(can_transition_to(&matrix, 0, 1));
    }
    #[test]
    fn communicates() {
        let matrix = Matrix3::from_vec(vec![0.5, 0.25, 0.0, 0.5, 0.5, 0.0, 0.0, 0.25, 1.0]);
        let graph = create_graph(&matrix, 3).unwrap();
        assert!(communicates_with(&graph, 0, 1));
        assert!(!communicates_with(&graph, 0, 2));
        assert!(communicates_with(&graph, 2, 2));
    }

    #[test]
    fn aperiodic() {
        let mut graph: Graph<usize, f64> = Graph::new();
        let one = graph.add_node(1);
        let two = graph.add_node(2);
        let three = graph.add_node(3);
        graph.extend_with_edges(&[
            (one, two),
            (one, one),
            (one, three),
            (two, three),
            (two, one),
            (two, two),
            (three, one),
        ]);
        assert_eq!(period(&graph, one), 1);
        assert_eq!(period(&graph, two), 1);
        assert_eq!(period(&graph, three), 1);
    }

    #[test]
    fn period_two() {
        let mut graph: Graph<usize, f64> = Graph::new();
        let one = graph.add_node(1);
        let two = graph.add_node(2);
        let three = graph.add_node(3);
        let four = graph.add_node(4);
        let five = graph.add_node(5);
        let six = graph.add_node(6);
        graph.extend_with_edges(&[
            (one, three),
            (three, five),
            (five, six),
            (six, four),
            (three, four),
            (four, two),
            (two, one),
        ]);
        assert_eq!(period(&graph, one), 2);
        assert_eq!(period(&graph, two), 2);
        assert_eq!(period(&graph, three), 2);
        assert_eq!(period(&graph, four), 2);
        assert_eq!(period(&graph, five), 2);
        assert_eq!(period(&graph, six), 2);
    }

    #[test]
    fn period_three() {
        let mut graph: Graph<usize, f64> = Graph::new();
        let one = graph.add_node(1);
        let two = graph.add_node(2);
        let three = graph.add_node(3);
        let four = graph.add_node(4);
        let five = graph.add_node(5);
        let six = graph.add_node(6);
        let seven = graph.add_node(7);
        graph.extend_with_edges(&[
            (one, three),
            (one, four),
            (one, five),
            (two, three),
            (two, five),
            (three, six),
            (three, seven),
            (four, six),
            (four, seven),
            (five, six),
            (five, seven),
            (six, one),
            (six, two),
            (seven, one),
            (seven, two),
        ]);
        assert_eq!(period(&graph, one), 3);
        assert_eq!(period(&graph, two), 3);
        assert_eq!(period(&graph, three), 3);
        assert_eq!(period(&graph, four), 3);
        assert_eq!(period(&graph, five), 3);
        assert_eq!(period(&graph, six), 3);
        assert_eq!(period(&graph, seven), 3);
    }
}
