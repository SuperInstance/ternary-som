# ternary-som

Self-organizing maps for ternary data — competitive learning, Gaussian neighborhood functions, ternary distance metrics, U-matrix visualization, topology preservation metrics, and codebook extraction over {-1, 0, +1}.

## Why This Exists

Self-organizing maps (SOMs) are powerful for dimensionality reduction and topology-preserving clustering, but standard implementations assume continuous real-valued input. When your data is inherently ternary — tri-state sensor readings, ternary logic outputs, sentiment-encoded features — you want a SOM that understands ternary distance and quantizes back to ternary values. This crate provides a rectangular SOM grid with competitive learning, Gaussian neighborhood decay, ternary-aware distance metrics, and quality measures (quantization error, topographic error). `forbid(unsafe_code)` throughout.

## Core Concepts

- **Trit**: Ternary value — `Neg` (-1), `Zero` (0), `Pos` (+1) with `to_f64()` and `from_f64()` quantization.
- **GridPos**: Row/column coordinates on the rectangular grid with Euclidean grid distance.
- **SomNode**: A neuron with continuous weight vectors and ternary-aware distance computation.
- **TernarySOM**: Rectangular SOM grid with `find_bmu()`, `train()`, `train_single()`, and configurable learning rate / sigma.
- **U-matrix**: Average distance to grid neighbors — reveals cluster boundaries.
- **Quality metrics**: `quantization_error()` (average BMU distance) and `topographic_error()` (fraction of inputs where BMU and second BMU aren't adjacent).
- **Codebook**: Extract quantized ternary weight vectors from all nodes.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-som = "0.1"
```

```rust
use ternary_som::{Trit, TernarySOM, GridPos};

fn main() {
    let data = vec![
        vec![Trit::Pos, Trit::Pos, Trit::Pos],
        vec![Trit::Neg, Trit::Neg, Trit::Neg],
        vec![Trit::Zero, Trit::Zero, Trit::Zero],
        vec![Trit::Pos, Trit::Zero, Trit::Neg],
        vec![Trit::Neg, Trit::Zero, Trit::Pos],
    ];

    // Create and train SOM
    let mut som = TernarySOM::new(3, 3, 3, 0.5, 1.5);
    som.train(&data, 50);
    som.decay(0.9); // reduce learning rate and sigma

    // Find best matching unit for a new input
    let bmu = som.find_bmu(&vec![Trit::Pos, Trit::Pos, Trit::Pos]);
    println!("BMU at row={}, col={}", bmu.row, bmu.col);

    // Map input to its BMU's quantized ternary weights
    let quantized = som.map_to_bmu(&vec![Trit::Pos, Trit::Pos, Trit::Pos]);

    // Quality metrics
    let qe = som.quantization_error(&data);
    let te = som.topographic_error(&data);
    println!("Quantization error: {:.4}", qe);
    println!("Topographic error: {:.4}", te);

    // U-matrix for visualization
    let umatrix = som.umatrix();
    for row in &umatrix {
        println!("{:?}", row);
    }

    // Extract full codebook
    let codebook = som.codebook();
}
```

## API Overview

| Type | Description |
|---|---|
| `Trit` | Ternary value: `Neg`, `Zero`, `Pos` |
| `GridPos` | Grid coordinates with `grid_distance()` |
| `SomNode` | Neuron with continuous weights, `ternary_distance()`, `quantized_weights()` |
| `TernarySOM` | Full SOM: `new()`, `find_bmu()`, `train()`, `train_single()`, `decay()` |
| `TernarySOM` metrics | `quantization_error()`, `topographic_error()`, `umatrix()`, `codebook()`, `map_to_bmu()` |

## How It Works

**TernarySOM** maintains a rectangular grid of `SomNode` instances, each with a continuous weight vector of dimension `dim`. Training follows the standard SOM algorithm:

1. **Find BMU**: For each input, compute the ternary distance (sum of squared differences between ternary input values projected to `{-1.0, 0.0, 1.0}` and the node's continuous weights). Select the node with minimum distance.
2. **Update weights**: Move each node's weights toward the input proportional to a Gaussian neighborhood function centered on the BMU: `Δw = η · h(bmu, node) · (input − w)`.
3. **Decay**: Reduce learning rate `η` and neighborhood width `σ` over time.

**Quantization** maps continuous weights back to ternary values via `from_f64()`: `< -0.33 → Neg`, `> 0.33 → Pos`, otherwise `Zero`. The **U-matrix** computes average weight-distance to 4-connected grid neighbors, highlighting cluster boundaries. **Topographic error** measures topology preservation by checking if the BMU and second-best BMU are adjacent on the grid.

## Use Cases

- **Ternary signal clustering**: Cluster tri-state sensor data while preserving topological relationships.
- **Dimensionality reduction for ternary features**: Map high-dimensional ternary vectors to a 2D grid for visualization.
- **Codebook generation**: Learn a compact ternary codebook from raw ternary data for compression or quantization.
- **Anomaly detection**: Identify inputs with high quantization error as outliers in ternary data streams.

## Known Limitations

- **Weight initialization is deterministic and input-independent**: `SomNode::new()` initializes weights as `((i × 0.1) % 3.0) − 1.0` for each dimension `i`. This means every SOM trained with the same grid dimensions starts with identical weights, regardless of the input data distribution. There is no random initialization or PCA-based seeding option.

- **No batch training shuffle**: `TernarySOM::train()` iterates the training data in the same order every epoch. Without shuffling, the SOM can develop directional biases — nodes toward the end of the training set's pattern distribution get preferentially mapped.

- **`from_f64()` quantization thresholds are hardcoded**: `Trit::from_f64()` uses fixed thresholds at ±0.33. These are not configurable and may not be optimal for all weight distributions. After many training iterations, weights can drift well outside [−1, 1], making the ±0.33 thresholds produce mostly `Neg` or `Pos` quantized values.

- **`topographic_error()` only checks 4-connectivity**: The BMU and second-BMU must be adjacent in the 4-connected sense (up/down/left/right). Diagonal neighbors are not considered adjacent, which can inflate the topographic error for small grids where diagonal adjacency matters.

- **`umatrix()` returns distances, not normalized values**: The U-matrix values are raw average distances to neighbors, which can vary widely in magnitude. There is no normalization option (min-max, z-score) — you must normalize externally for visualization.

- **`decay()` applies multiplicative decay with no floor**: Calling `decay(0.9)` multiplies learning rate and sigma by 0.9. After many decays, sigma can approach 0, making the neighborhood function so narrow that only the BMU itself is updated — equivalent to k-means with no smoothing.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — gradient-free optimization on ternary landscapes
- `ternary-language` — ternary NLP and grammar processing
- `ternary-trees` — ternary decision trees and forests
- `ternary-transform` — wavelet, Fourier, and kernel transforms
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — reinforcement learning with ternary actions
- `ternary-som` — this crate
- `ternary-failure` — failure analysis with ternary classification

## License

MIT
