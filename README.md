# BARI2D

BARI2D is a research framework for decentralized multi-robot bridge construction. Twenty identical rectangular robots use one shared recurrent policy, local IR and strain histories, local internal state, and a mission-level target load. A centralized critic may use simulator state during training, but the actor never receives the map, robot positions, contact graph, current bridge capacity, or other global structural information.

The framework currently implements the complete 2D/2.5D environment and structural-evaluation phases plus a recurrent parameter-sharing MAPPO baseline. No bridge topology, truss pattern, preferred angle, or geometry template appears in the controller or reward.

## Installation

The working environment for this repository is the existing `bari2d` conda environment:

```bash
conda activate bari2d
python -m pip install -e '.[dev]'
pytest
```

All commands in this project should run inside that environment.

## Quick start

Train the canonical capacity-teacher/local-policy system:

```bash
conda run -n bari2d python scripts/train.py \
  --config configs/capacity_local.yaml \
  --updates 2000
```

Resume training:

```bash
conda run -n bari2d python scripts/train.py \
  --resume runs/capacity_local/checkpoint_002000.pt \
  --updates 3000
```

Evaluate with the exact 6-DOF capacity solver:

```bash
conda run -n bari2d python scripts/evaluate.py \
  runs/capacity_local/checkpoint_002000.pt \
  --config configs/capacity_local.yaml --stage 4 --episodes 100
```

The old YAML actor/critic ablation suite has been removed. `capacity_local.yaml` is the canonical policy-training setup.

## Simulator

`BridgeEnv` exposes a Gym-style `reset`/`step` interface without requiring Gymnasium. Each robot is an oriented rectangle with position, heading, scalar forward velocity, steering, head-lift state, anchor state, local strain, previous action, layer, and optional persistent latent. Planar motion uses bounded kinematics. Same-layer overlap produces a normal positional response, frictional velocity damping, and a strain-producing contact force. A climb command moves a robot one layer above a nearby same-layer robot. An elevated robot remains there only while its footprint overlaps a robot on the layer directly below; otherwise it falls to one layer above the highest overlapping lower robot, or to layer 0 when none overlaps. This support rule does not use moments or exempt anchored robots. Robots unsupported over the gap fall out of the active mechanical structure.

The field uses coordinates relative to a randomized gap normal. It supports gap width, orientation, sinusoidal boundary irregularity, bank shape, random initial poses, friction, anchor strength, mass, sensor noise, and actuator noise. Generated widths are capped to remain feasible for the configured field and robot population.

Anchors are persistent robot-robot or robot-bank edges. Configurable tension, compression, shear, rotational stiffness, and stochastic failure values determine their dynamic behavior. Excessive extension or shear breaks an anchor. Contact and bank edges remain transient.

### Local observation

Every continuous value is normalized. One actor observation contains:

- IR history from configurable front, back, left, right, and downward rays;
- local strain history;
- previous-action history;
- anchor state, head-lift state, velocity, steering, height layer, and fallen state;
- normalized target load;
- optional episode-persistent Gaussian latent.

The four planar IR rays ray-march against robot oriented boxes, circular obstacles, field limits, and bank/gap substrate transitions. The downward IR ray returns the vertical distance to a bank under the robot or to the highest lower-layer robot whose oriented footprint has positive-area overlap; maximum range means there is no supporting surface below (a cliff).

The centralized state and graph are separate methods used only by critics and diagnostics. Tests and model construction preserve this actor/critic boundary.

### Discrete action space

The baseline uses the permitted fully discrete fallback:

1. forward
2. backward
3. forward-left
4. forward-right
5. backward-left
6. backward-right
7. climb
8. anchor
9. release
10. idle

Masks remove only physically impossible actions: climb without nearby support, anchor without a mechanical partner, release while unanchored, motion while anchored, and all non-idle actions after falling. Masks do not encode behavioral or structural rules.

### Contact graph and capacity

The mechanical graph contains robot, left-bank, and right-bank nodes. Edges represent active contact, anchors, or bank support. Breadth-first connectivity supplies the binary span indicator. Continuous progress is the furthest left-connected robot measured across the local gap width.

Two replaceable evaluators implement structural scoring:

- `FastLoadEvaluator` uses graph maximum flow and path compliance on every construction step.
- `IncrementalLoadEvaluator` freezes the morphology and raises external load in configurable increments until contact loss, anchor failure, robot slip, excessive displacement/structural collapse, or loss of connectivity.

Both use a modular load-protocol strategy. `center_point` applies demand at the robot nearest the geometric center of the central spanning component. The default `uniform` protocol distributes demand across all independent spanning components and combines their capacities and stiffnesses. The interface is ready for moving, random-position, or multi-point load implementations.

This is a research proxy, not a validated structural-engineering solver. Edge capacity and compliance are meaningful and monotonic under added parallel load paths, but they do not reproduce continuum mechanics.

### Reward

Policy learning now uses a training-only structural capacity teacher. Construction runs for a fixed horizon; reaching an old scalar target-load proxy no longer terminates the episode early. At the final state `G_T`, the teacher computes or predicts the physically defined payload capacity `C(G_T)` and turns it into the main terminal team return.

The teacher also computes leave-one-robot-out counterfactual capacities `C(G_T \ i)`. Their differences provide per-robot structural credit to MAPPO. Small progress, energy, anchor, and collapse terms remain only as exploration/efficiency shaping in `configs/capacity_local.yaml`; the legacy graph-flow mechanical reward is set to zero.

The actor never observes capacity, counterfactual credit, the global graph, or the teacher output. These signals exist only during centralized training.

## Policy and critics

The canonical deployed policy is one shared GRU. Every robot receives only its own local observation/history and maintains its own recurrent hidden state. There is no robot ID, global position, contact graph, capacity prediction, or centralized message in the actor input. The same weights therefore implement one decentralized local rule on every robot.

During training only, the graph critic receives the global contact state and estimates the expected future terminal return. The capacity teacher is also training-only. Neither is required for inference.

Older actor classes remain in the source for checkpoint compatibility, but the old YAML ablation suite is no longer part of the training workflow.

## MAPPO

The trainer collects team rewards with per-agent actions and log probabilities, computes GAE, and applies clipped PPO updates. Agent sequences remain time-major during updates so the shared GRU receives backpropagation through time; reset masks clear hidden state at episode boundaries. The centralized value loss and branch-specific auxiliary losses are optimized separately from decentralized action selection.

Robot positions and headings are randomized on every reset. Gap width, orientation, and boundary irregularity are randomized at every curriculum stage, with the sampled range expanding as difficulty increases. Curriculum stage advances when a rolling success window reaches the configured threshold:

1. 25% of the configured terrain-randomization range and low load;
2. 50% of the terrain range and randomized target load;
3. 75% of the terrain range;
4. friction, mass, anchor strength, sensor noise, and actuator noise randomization.

## Outputs

`runs/<name>/episodes.jsonl` stores success, gap parameters, target and measured capacity, capacity ratio, construction time, used and anchored robot counts, energy proxy, anchor failures, fallen count, maximum progress, final contact graph, final morphology, action distribution, and available branch-latent statistics. Checkpoints include actor, critic, optimizers, configuration, and update number.

Visualization shows terrain, gap, robot rectangles and orientation, layer color, anchored edges, contact graph, planar and downward IR markers, target load, capacity, progress, and any load-bearing path returned by the terminal evaluator. `record_episode` can save a GIF for any callable policy.

## Verification

```bash
conda run -n bari2d pytest
```

Tests cover gap generation, movement and steering, IR detection, contact and climbing, anchoring and failure, graph connectivity, bank-to-bank spanning, fast and terminal load evaluation, monotonic strength of parallel structures, reward invariants, every actor ablation, both critics, and a full MAPPO update/checkpoint.

## Assumptions and current limits

- Height is a discrete layer, not full 3D rigid-body dynamics.
- Contact response is positional and quasi-static; it has no angular impulse solver.
- Structural capacity is a graph-flow/compliance approximation with incremental failure checks.
- One anchor is owned per robot. An anchor may attach to a bank or another robot.
- Robots share the target-load scalar as mission context; all other deployment observations are local.
- Evaluation generalization means held-out random seeds and stage-4 parameter distributions, not arbitrary out-of-distribution terrain.
- Learned morphology quality depends on training budget and hyperparameters. This repository provides a functioning research baseline, not pretrained evidence of convergence.

The structural evaluator, load protocol, dynamics, actor, and critic are separate modules so higher-fidelity physics or alternative learning algorithms can replace them without exposing global state to the decentralized actor.


## Diagram Overview

[![Architecture diagram of whishingraven/bari2dsandbox](https://gitdiagram.com/whishingraven/bari2dsandbox/diagram.png)](https://gitdiagram.com/whishingraven/bari2dsandbox?utm_source=readme&utm_medium=picture)

### 6-DOF static bridge capacity

The optional mechanics layer can evaluate the current 2.5-D morphology as an
extruded 3-D rigid-body structure and report the contact wrench
`(Fx, Fy, Fz, Mx, My, Mz)`. Each contact also exposes the connector-local
`(Axial, ShearU, ShearV, Torsion, BendingU, BendingV)` representation. It
includes robot self-weight and can compute the first connector-local strength
limit reached by an added payload analytically.
See `docs/spatial_statics.md`.

## Procedural wrench-capacity learning

The repository also contains a supervised structural-learning pipeline built on the 6-DOF spatial statics solver. `RandomBridgeGenerator` creates unlimited variable-size bank-to-bank morphologies with parallel rows, braces, branches, and stacked robots. `CapacityDatasetBuilder` samples six-component connector-local strength limits, applies a 1 kg gravity payload at a bridge-deck location, includes robot self-weight, and computes the exact first contact/component failure mass.

Generate a finite dataset:

```bash
python scripts/generate_capacity_dataset.py \
  --samples 100000 --workers 4 \
  --feature-mode mechanics_informed \
  --payload-mode random \
  --output runs/capacity/mechanics_100k.pt
```

Train from a finite dataset:

```bash
python scripts/train_capacity_gnn.py \
  --dataset runs/capacity/mechanics_100k.pt \
  --epochs 50 --batch-size 64 \
  --output runs/capacity/mechanics_gnn.pt
```

Or generate fresh bridges indefinitely during training:

```bash
python scripts/train_capacity_gnn.py \
  --feature-mode geometry_only \
  --steps 20000 --batch-size 32 --workers 4 \
  --output runs/capacity/geometry_online.pt
```

See `docs/capacity_learning.md` for the graph features, exact labeling equation, auxiliary failure-location targets, and the distinction between the mechanics-informed control baseline and the scientifically harder geometry-only prediction task.
