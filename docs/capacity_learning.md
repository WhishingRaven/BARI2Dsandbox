# Wrench-limited bridge capacity learning

This module turns the 6-DOF spatial statics solver into a procedural supervised-learning pipeline.

## Definition of capacity

Each physical connection has its own connector frame `(n, t_u, t_v)`. The
strength calculation uses the local wrench

`w_c = (Axial, ShearU, ShearV, Torsion, BendingU, BendingV)`.

The simulator is still allowed to keep every existing connection intact.
Strength is evaluated separately with component-wise local limits

`|w_c,k| <= w_max,c,k`.

With linear small-displacement statics, the contact wrench under a payload of
mass `m` is

`w_c(m) = w_c,self + m * w_c,1kg`.

The ground-truth target is the first positive mass for which any local component
of any connection reaches its limit. Robot self-weight is included by default.

This is preferable to world-axis limits because connector axial/shear/bending/
torsion strength remains attached to the connector when the bridge rotates.

## Procedural bridge generator

`bari2d.bridge_generation.RandomBridgeGenerator` generates an unlimited stream
of variable-size structures. It first creates one or more guaranteed bank-to-bank rows and then randomizes:

- gap width and gap orientation;
- number of parallel rows;
- robot spacing, lateral jitter, and yaw;
- cross braces and diagonal braces;
- side branches whose self-weight loads the main structure;
- stacked robots and upper-layer links, which create nonzero 3-D lever arms.

The bridge is represented by the existing `RobotState`, `GapField`, and
`ContactGraph` classes, so the exact same spatial statics solver is used for
labeling.

## Graph features

The graph includes every robot plus virtual left-bank and right-bank nodes.
Every physical connection becomes two directed message-passing edges.

Node features contain node type, pose in gap-relative coordinates, relative yaw,
layer, payload location, and robot mass.

Edge geometry contains:

- connection type;
- relative geometry and contact-point offsets;
- relative yaw;
- scalar stiffness;
- the contact-frame axes `(n, t_u, t_v)` expressed in gap-relative coordinates;
- the six local strength limits: axial, two shear, torsion and two bending limits.

Two feature modes are provided:

- `mechanics_informed`: also includes `w_self / w_max` and `w_1kg / w_max` for all six local components. This is the recommended pipeline-validation baseline.
- `geometry_only`: omits the solved wrench responses. The GNN must infer load transfer from topology, geometry, contact-frame orientation, stiffness, and strength limits. This is the harder research task.

The current feature schema is version 2 and records
`wrench_frame="contact_local"` in sample metadata. Version-1 checkpoints/datasets
used world-axis wrench limits and have a different edge-feature width.

The payload robot is sampled on the bridge deck in `random` mode or selected
nearest the gap center in `center` mode.

## Offline dataset generation

```bash
python scripts/generate_capacity_dataset.py \
  --samples 100000 \
  --feature-mode mechanics_informed \
  --payload-mode random \
  --output runs/capacity/mechanics_100k.pt
```

For the harder ablation:

```bash
python scripts/generate_capacity_dataset.py \
  --samples 100000 \
  --feature-mode geometry_only \
  --output runs/capacity/geometry_100k.pt
```

## Infinite online generation

`InfiniteBridgeDataset` is a PyTorch `IterableDataset`. It generates a fresh
bridge, fresh connector-local strength limits, a fresh payload location, and a
fresh statics label on every iteration. Therefore training is not limited to a
fixed list of bridge topologies.

## Training

Finite dataset:

```bash
python scripts/train_capacity_gnn.py \
  --dataset runs/capacity/mechanics_100k.pt \
  --epochs 50 \
  --batch-size 64 \
  --output runs/capacity/mechanics_gnn.pt
```

Infinite procedural training:

```bash
python scripts/train_capacity_gnn.py \
  --feature-mode mechanics_informed \
  --steps 20000 \
  --batch-size 32 \
  --output runs/capacity/online_gnn.pt
```

The network is dependency-free sparse message passing implemented with PyTorch
`index_add_`; PyTorch Geometric is not required. It predicts:

1. `log(1 + maximum payload kg)` for robust scalar regression;
2. the critical local wrench component (`Axial`, `ShearU`, `ShearV`, `Torsion`, `BendingU`, or `BendingV`);
3. the critical graph connection.

The auxiliary failure-location tasks are useful diagnostics: a low scalar error
with random failure localization is evidence that the model may be exploiting
dataset correlations rather than learning the structural load path.

## Important interpretation

In `mechanics_informed` mode, `w_1kg` makes the exact capacity analytically
recoverable. The GNN in that mode should therefore be treated as a
pipeline/control baseline, not as evidence that the network discovered
mechanics. The scientifically stronger comparison is:

1. exact statics solver (ground truth),
2. mechanics-informed GNN,
3. geometry-only GNN,
4. geometry-only GNN evaluated on bridge sizes/topologies outside the training distribution.

## Per-contact auxiliary supervision

In addition to the graph-level maximum payload, every directed graph edge stores
the corresponding physical connection's own failure mass (clipped only for
otherwise-unbounded edges). The model predicts these edge capacities and uses a
differentiable soft-min to obtain the bridge-level capacity. This matches the
governing definition: the bridge fails when the first contact reaches a local
connector limit.

For procedurally generated bridges, the statics code uses a fast direct-solve
path because the generator guarantees welded 6-DOF paths to fixed supports.
General `SpatialStaticSolver` calls retain rank verification by default. Dataset
generation can use multiple processes through `--workers`.

## Paired graph / top-down bridge visualization

To display the **same bridge** as a node-link graph and as the top-down BARI2D
simulator scene (real robot rectangles, yaw, layer overlap, bank terrain), use:

```bash
python scripts/evaluate_capacity_gnn.py runs/capacity/mechanics_gnn.pt \
  --samples 200 --seed 20000 --output-dir runs/capacity/evaluation
```

In addition to the existing prediction figures, this now saves:

- `04_representative_bridge_dual_view.png`
- `05_worst_overprediction_bridge_dual_view.png`

The left panel is the **GNN graph representation**. The right panel depicts
**physical top-view robot footprints** in the same gap-aligned coordinates.
Exact critical contacts are marked in red; predicted critical contacts in blue;
payload is starred. Robot layers overlap in this top-down projection, as they do
in the underlying 2.5-D environment.

If only the old graph PNG files remain (no model checkpoint), reconstruct the
same bridges from their procedural evaluation indices:

```bash
python scripts/render_capacity_topdown.py \
  --seed 20000 --indices 73 105 \
  --graph-images 04_representative_bridge.png 05_worst_overprediction_bridge.png \
  --output-dir runs/capacity/topdown
```

This exact reconstruction assumes the **same generator code/version**, same
single-worker seed and sample order. The script regenerates labels but does not
recompute GNN predictions. For old datasets without `topdown_view` metadata,
robot poses are reconstructed from node features with default robot dimensions;
new datasets retain exact payload and contact point locations as optional
`topdown_view` metadata without changing tensor feature dimensions or schema.

The paired `*_dual_view.png` and single `*_topdown.png` now label robots as
`graph node ID / L{layer}` (for example `12 / L2`), using actual integer
`RobotState.layer`, not the normalized feature. A per-layer robot ID roster is
printed beneath paired figures so even robots occluded in a top-down view
remain identifiable. Layer 0 is the base layer. Different layers use distinct
body/node colors; GNN edge colors still encode predicted failure capacity.
The optional metadata `topdown_view.robot_layers` stores exact layer integers,
while legacy saved graph samples recover them from `z / robot_height` in node
feature index 5. The input tensor shapes and model checkpoint schema are unchanged.
