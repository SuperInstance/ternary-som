#![forbid(unsafe_code)]

//! Self-organizing maps (SOM) for ternary data.
//!
//! Provides TernarySOM grid training with competitive learning, ternary distance
//! metrics, neighborhood functions, weight adaptation on {-1,0,+1}, topology
//! preservation metrics, and U-matrix visualization data.

use std::fmt;

/// A trit value in {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg,
    Zero,
    Pos,
}

impl Trit {
    pub fn to_f64(self) -> f64 {
        match self {
            Trit::Neg => -1.0,
            Trit::Zero => 0.0,
            Trit::Pos => 1.0,
        }
    }

    pub fn from_f64(v: f64) -> Self {
        if v < -0.33 {
            Trit::Neg
        } else if v > 0.33 {
            Trit::Pos
        } else {
            Trit::Zero
        }
    }
}

/// Grid coordinates for a SOM node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridPos {
    pub row: usize,
    pub col: usize,
}

impl GridPos {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    /// Euclidean distance on the grid.
    pub fn grid_distance(&self, other: &GridPos) -> f64 {
        let dr = self.row as f64 - other.row as f64;
        let dc = self.col as f64 - other.col as f64;
        (dr * dr + dc * dc).sqrt()
    }
}

/// A single node (neuron) in the SOM, holding continuous weight vectors.
#[derive(Debug, Clone)]
pub struct SomNode {
    pub pos: GridPos,
    pub weights: Vec<f64>,
}

impl SomNode {
    pub fn new(pos: GridPos, dim: usize) -> Self {
        let mut weights = Vec::with_capacity(dim);
        // Initialize near zero for ternary data
        for i in 0..dim {
            weights.push(((i as f64 * 0.1) % 3.0) - 1.0);
        }
        Self { pos, weights }
    }

    /// Ternary distance: sum of squared differences between input and weights,
    /// with ternary inputs projected to f64.
    pub fn ternary_distance(&self, input: &[Trit]) -> f64 {
        self.weights
            .iter()
            .zip(input.iter())
            .map(|(&w, &t)| {
                let diff = w - t.to_f64();
                diff * diff
            })
            .sum()
    }

    /// Euclidean distance between weight vectors.
    pub fn weight_distance(&self, other: &SomNode) -> f64 {
        self.weights
            .iter()
            .zip(other.weights.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Quantize weights back to ternary values.
    pub fn quantized_weights(&self) -> Vec<Trit> {
        self.weights.iter().map(|&w| Trit::from_f64(w)).collect()
    }
}

/// A rectangular self-organizing map for ternary data.
pub struct TernarySOM {
    pub rows: usize,
    pub cols: usize,
    pub dim: usize,
    pub nodes: Vec<Vec<SomNode>>,
    pub learning_rate: f64,
    pub sigma: f64,
}

impl TernarySOM {
    /// Create a new SOM with the given grid size and input dimension.
    pub fn new(rows: usize, cols: usize, dim: usize, learning_rate: f64, sigma: f64) -> Self {
        let nodes = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| SomNode::new(GridPos::new(r, c), dim))
                    .collect()
            })
            .collect();
        Self {
            rows,
            cols,
            dim,
            nodes,
            learning_rate,
            sigma,
        }
    }

    /// Find the Best Matching Unit (BMU) for a given input.
    pub fn find_bmu(&self, input: &[Trit]) -> GridPos {
        let mut best_pos = GridPos::new(0, 0);
        let mut best_dist = f64::MAX;
        for row in &self.nodes {
            for node in row {
                let d = node.ternary_distance(input);
                if d < best_dist {
                    best_dist = d;
                    best_pos = node.pos;
                }
            }
        }
        best_pos
    }

    /// Gaussian neighborhood function.
    pub fn neighborhood(&self, bmu: &GridPos, pos: &GridPos) -> f64 {
        let dist = bmu.grid_distance(pos);
        (-dist * dist / (2.0 * self.sigma * self.sigma)).exp()
    }

    /// Train the SOM on a single input sample.
    pub fn train_single(&mut self, input: &[Trit]) {
        assert_eq!(input.len(), self.dim);
        let bmu = self.find_bmu(input);
        let lr = self.learning_rate;
        let sigma = self.sigma;
        for row in &mut self.nodes {
            for node in row.iter_mut() {
                let dist = bmu.grid_distance(&node.pos);
                let h = (-dist * dist / (2.0 * sigma * sigma)).exp();
                let input_vals: Vec<f64> = input.iter().map(|t| t.to_f64()).collect();
                for (w, &iv) in node.weights.iter_mut().zip(input_vals.iter()) {
                    *w += lr * h * (iv - *w);
                }
            }
        }
    }

    /// Train the SOM on a dataset for the given number of epochs.
    pub fn train(&mut self, data: &[Vec<Trit>], epochs: usize) {
        for _ in 0..epochs {
            for input in data {
                self.train_single(input);
            }
        }
    }

    /// Decay learning rate and sigma over time.
    pub fn decay(&mut self, factor: f64) {
        self.learning_rate *= factor;
        self.sigma *= factor;
        if self.sigma < 0.1 {
            self.sigma = 0.1;
        }
    }

    /// Get the node at a specific position.
    pub fn get_node(&self, pos: &GridPos) -> &SomNode {
        &self.nodes[pos.row][pos.col]
    }

    /// Get a mutable reference to the node at a specific position.
    pub fn get_node_mut(&mut self, pos: &GridPos) -> &mut SomNode {
        &mut self.nodes[pos.row][pos.col]
    }

    /// Quantization error: average distance from each input to its BMU.
    pub fn quantization_error(&self, data: &[Vec<Trit>]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let total: f64 = data
            .iter()
            .map(|input| {
                let bmu_pos = self.find_bmu(input);
                self.get_node(&bmu_pos).ternary_distance(input).sqrt()
            })
            .sum();
        total / data.len() as f64
    }

    /// Topographic error: fraction of inputs where BMU and second-best BMU are not adjacent.
    pub fn topographic_error(&self, data: &[Vec<Trit>]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut errors = 0usize;
        for input in data {
            let mut dists: Vec<(f64, GridPos)> = Vec::new();
            for row in &self.nodes {
                for node in row {
                    dists.push((node.ternary_distance(input), node.pos));
                }
            }
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if dists.len() >= 2 {
                let d = dists[0].1.grid_distance(&dists[1].1);
                if d > 1.5 {
                    errors += 1;
                }
            }
        }
        errors as f64 / data.len() as f64
    }

    /// Compute U-matrix: for each node, average distance to its neighbors.
    pub fn umatrix(&self) -> Vec<Vec<f64>> {
        let mut u = vec![vec![0.0; self.cols]; self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let node = &self.nodes[r][c];
                let mut sum = 0.0;
                let mut count = 0usize;
                // 4-connected neighbors
                let neighbors = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
                for (dr, dc) in &neighbors {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                        sum += node.weight_distance(&self.nodes[nr as usize][nc as usize]);
                        count += 1;
                    }
                }
                u[r][c] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }
        u
    }

    /// Return the quantized weight vectors for all nodes as a flat list.
    pub fn codebook(&self) -> Vec<Vec<Trit>> {
        self.nodes
            .iter()
            .flat_map(|row| row.iter().map(|n| n.quantized_weights()))
            .collect()
    }

    /// Map an input to its BMU's quantized (ternary) weight vector.
    pub fn map_to_bmu(&self, input: &[Trit]) -> Vec<Trit> {
        let bmu = self.find_bmu(input);
        self.get_node(&bmu).quantized_weights()
    }
}

impl fmt::Debug for TernarySOM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TernarySOM({}x{}, dim={})", self.rows, self.cols, self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> Vec<Vec<Trit>> {
        vec![
            vec![Trit::Pos, Trit::Pos, Trit::Pos],
            vec![Trit::Neg, Trit::Neg, Trit::Neg],
            vec![Trit::Zero, Trit::Zero, Trit::Zero],
            vec![Trit::Pos, Trit::Zero, Trit::Neg],
            vec![Trit::Neg, Trit::Zero, Trit::Pos],
            vec![Trit::Zero, Trit::Pos, Trit::Zero],
            vec![Trit::Pos, Trit::Pos, Trit::Zero],
            vec![Trit::Neg, Trit::Neg, Trit::Zero],
        ]
    }

    #[test]
    fn test_trit_to_from_f64() {
        assert_eq!(Trit::from_f64(Trit::Neg.to_f64()), Trit::Neg);
        assert_eq!(Trit::from_f64(Trit::Zero.to_f64()), Trit::Zero);
        assert_eq!(Trit::from_f64(Trit::Pos.to_f64()), Trit::Pos);
        assert_eq!(Trit::from_f64(-0.5), Trit::Neg);
        assert_eq!(Trit::from_f64(0.0), Trit::Zero);
        assert_eq!(Trit::from_f64(0.5), Trit::Pos);
    }

    #[test]
    fn test_grid_distance() {
        let a = GridPos::new(0, 0);
        let b = GridPos::new(3, 4);
        assert!((a.grid_distance(&b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_grid_distance_same() {
        let a = GridPos::new(2, 3);
        assert!(a.grid_distance(&a).abs() < 0.001);
    }

    #[test]
    fn test_som_creation() {
        let som = TernarySOM::new(3, 4, 2, 0.5, 1.0);
        assert_eq!(som.rows, 3);
        assert_eq!(som.cols, 4);
        assert_eq!(som.dim, 2);
    }

    #[test]
    fn test_find_bmu() {
        let som = TernarySOM::new(3, 3, 3, 0.1, 1.0);
        let input = vec![Trit::Pos, Trit::Pos, Trit::Pos];
        let bmu = som.find_bmu(&input);
        assert!(bmu.row < 3);
        assert!(bmu.col < 3);
    }

    #[test]
    fn test_neighborhood_center() {
        let som = TernarySOM::new(5, 5, 2, 0.1, 1.0);
        let bmu = GridPos::new(2, 2);
        let nh = som.neighborhood(&bmu, &bmu);
        assert!((nh - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_neighborhood_falloff() {
        let som = TernarySOM::new(5, 5, 2, 0.1, 1.0);
        let bmu = GridPos::new(0, 0);
        let near = som.neighborhood(&bmu, &GridPos::new(0, 1));
        let far = som.neighborhood(&bmu, &GridPos::new(4, 4));
        assert!(near > far);
    }

    #[test]
    fn test_train_single_changes_weights() {
        let mut som = TernarySOM::new(2, 2, 3, 0.5, 1.0);
        let weights_before: Vec<f64> = som.nodes[0][0].weights.clone();
        som.train_single(&vec![Trit::Pos, Trit::Pos, Trit::Pos]);
        let weights_after: Vec<f64> = som.nodes[0][0].weights.clone();
        assert_ne!(weights_before, weights_after);
    }

    #[test]
    fn test_train_epochs() {
        let mut som = TernarySOM::new(3, 3, 3, 0.5, 1.5);
        let data = make_data();
        som.train(&data, 10);
        // After training, quantization error should be finite
        let qe = som.quantization_error(&data);
        assert!(qe.is_finite());
    }

    #[test]
    fn test_quantization_error() {
        let som = TernarySOM::new(3, 3, 3, 0.1, 1.0);
        let data = make_data();
        let qe = som.quantization_error(&data);
        assert!(qe >= 0.0);
    }

    #[test]
    fn test_quantization_error_empty() {
        let som = TernarySOM::new(2, 2, 2, 0.1, 1.0);
        assert_eq!(som.quantization_error(&[]), 0.0);
    }

    #[test]
    fn test_topographic_error() {
        let mut som = TernarySOM::new(5, 5, 3, 0.5, 2.0);
        let data = make_data();
        som.train(&data, 50);
        let te = som.topographic_error(&data);
        assert!(te >= 0.0 && te <= 1.0);
    }

    #[test]
    fn test_topographic_error_empty() {
        let som = TernarySOM::new(2, 2, 2, 0.1, 1.0);
        assert_eq!(som.topographic_error(&[]), 0.0);
    }

    #[test]
    fn test_umatrix() {
        let som = TernarySOM::new(3, 4, 2, 0.1, 1.0);
        let u = som.umatrix();
        assert_eq!(u.len(), 3);
        assert_eq!(u[0].len(), 4);
        for row in &u {
            for &val in row {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_umatrix_single_node() {
        let som = TernarySOM::new(1, 1, 2, 0.1, 1.0);
        let u = som.umatrix();
        assert_eq!(u[0][0], 0.0);
    }

    #[test]
    fn test_codebook() {
        let som = TernarySOM::new(2, 3, 3, 0.1, 1.0);
        let cb = som.codebook();
        assert_eq!(cb.len(), 6); // 2x3
        for v in &cb {
            assert_eq!(v.len(), 3);
            for t in v {
                assert!(matches!(t, Trit::Neg | Trit::Zero | Trit::Pos));
            }
        }
    }

    #[test]
    fn test_map_to_bmu() {
        let mut som = TernarySOM::new(3, 3, 3, 0.5, 1.5);
        let data = make_data();
        som.train(&data, 20);
        let result = som.map_to_bmu(&vec![Trit::Pos, Trit::Pos, Trit::Pos]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_decay() {
        let mut som = TernarySOM::new(2, 2, 2, 0.5, 2.0);
        som.decay(0.9);
        assert!((som.learning_rate - 0.45).abs() < 0.001);
        assert!((som.sigma - 1.8).abs() < 0.001);
    }

    #[test]
    fn test_decay_sigma_floor() {
        let mut som = TernarySOM::new(2, 2, 2, 0.1, 0.11);
        som.decay(0.5);
        assert!((som.sigma - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_quantized_weights() {
        let node = SomNode {
            pos: GridPos::new(0, 0),
            weights: vec![-0.8, 0.1, 0.9],
        };
        let q = node.quantized_weights();
        assert_eq!(q, vec![Trit::Neg, Trit::Zero, Trit::Pos]);
    }

    #[test]
    fn test_ternary_distance() {
        let node = SomNode {
            pos: GridPos::new(0, 0),
            weights: vec![1.0, 0.0, -1.0],
        };
        let input = vec![Trit::Pos, Trit::Zero, Trit::Neg];
        let d = node.ternary_distance(&input);
        assert!(d.abs() < 0.001);
    }

    #[test]
    fn test_debug_format() {
        let som = TernarySOM::new(3, 4, 5, 0.1, 1.0);
        let s = format!("{:?}", som);
        assert!(s.contains("TernarySOM"));
    }
}
