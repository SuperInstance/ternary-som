# Future Integration: ternary-som

## Current State
Provides self-organizing maps (SOM) for ternary data: competitive learning, ternary distance metrics, neighborhood functions, weight adaptation on {-1, 0, +1}, topology preservation metrics, and U-matrix visualization data.

## Integration Opportunities

### With ternary-cell (Feature Clustering)
Cell grids produce high-dimensional state vectors. `ternary-som` clusters these into a 2D map where similar states are nearby. Cells that are neighbors on the SOM grid are functionally similar, even if they're distant in the actual grid. This reveals hidden structure: cells cluster into types (explorers, stabilizers, specialists) based on behavior, not position.

### With ternary-visualization
U-matrix data from the SOM IS visualization data. The U-matrix shows cluster boundaries as peaks — easy to render as a heatmap. `ternary-visualization` renders the SOM; `ternary-som` computes it. Together: real-time visualization of room state clustering.

### With ternary-clustering
`ternary-som` IS a clustering algorithm, but topology-preserving: it maintains the spatial structure of the clusters. `ternary-clustering` provides non-topological clustering (k-means, hierarchical). Use SOM for visualization and navigation, k-means for pure classification.

## Potential in Mature Systems
In room-as-codespace, the SOM organizes the room campus. Rooms that are functionally similar cluster together on the map. An agent navigating the campus uses the SOM as a guide: "I need a room like this one, show me nearby rooms on the SOM." U-matrix boundaries indicate room type transitions — crossing a boundary means entering a different functional area.

## Cross-Pollination Ideas
- SOM as a room catalog — agents query the map to find rooms matching desired characteristics
- Neighborhood function as room influence radius — when a room changes, how far does the effect propagate?
- Topology preservation as a measure of room organization quality

## Dependencies for Next Steps
- ternary-cell needs feature extraction for SOM input
- Integration with ternary-visualization for U-matrix rendering
- Integration with ternary-clustering for combined analysis
