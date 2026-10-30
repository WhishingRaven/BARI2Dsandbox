# Capacity-teacher decentralized policy training

## Goal

The deployed controller is a shared recurrent local policy. Robot `i` chooses

\[
a_i(t) \sim \pi_\theta(a_i\mid o_i(t),h_i(t))
\]

from its own sensor/history state only. It never receives the global map, robot positions, contact graph, bridge capacity, the centralized critic state, or the capacity teacher output.

Training uses centralized training / decentralized execution (CTDE). Global structural information is allowed only on the training side.

## Fixed construction horizon

`environment.episode_objective: capacity_horizon` disables the old rule that ended an episode as soon as a scalar target-load proxy was reached. Construction continues until `max_steps`. This matters because a spanning bridge may still be strengthened by adding braces or anchors.

At the horizon the capacity teacher evaluates the final structure.

## Capacity terminal return

For the completed morphology `G_T`, the teacher obtains a payload capacity

\[
C(G_T)\;[\mathrm{kg}].
\]

The default teacher uses the exact 6-DOF contact-local statics solver. A trained geometry-only `CapacityGNN` can be used as a faster surrogate.

The global terminal reward uses a bounded logarithmic transform

\[
r_T^{cap}=\alpha\log\left(1+\frac{C(G_T)}{C_{ref}}\right).
\]

A non-spanning structure receives `no_span_penalty`.

The old graph-flow mechanical-capacity reward is disabled in `configs/capacity_local.yaml`; only small progress/efficiency shaping remains. Structural quality is therefore supplied by the capacity teacher.

## Counterfactual robot credit

For every robot in the final spanning structural component, training evaluates a leave-one-robot-out counterfactual

\[
C_{-i}=C(G_T\setminus i).
\]

The agent-specific structural credit is

\[
d_i = s(C(G_T))-s(C_{-i}),\qquad
s(C)=\log\left(1+\frac{C}{C_{ref}}\right).
\]

If removing robot `i` destroys the span, `C_{-i}=0`. If removing a useless/heavy branch improves capacity, `d_i` can be negative. Credits are clipped before being stored in `agent_rewards`.

MAPPO combines the shared global advantage with this robot-specific advantage:

\[
A_{t,i}=A_t^{global}+\lambda_{credit}A_{t,i}^{counterfactual}.
\]

The purpose is credit assignment only. The policy still has no access to the counterfactual calculation at execution time.

## Centralized graph critic

The canonical config uses the existing training-only `GraphCritic`. Its job is now especially important: it learns the expected terminal capacity return from partial construction states and propagates the sparse terminal signal backward through GAE.

In other words, the critic acts as the learned construction value function

\[
V(S_t)\approx \mathbb E[C(G_T)\mid S_t],
\]

while the actor is restricted to local observations.

## Capacity teacher backends

### Exact

```yaml
training:
  capacity_teacher:
    backend: exact
```

Uses `SpatialStaticSolver.capacity_local(...)`. This is the ground-truth training/evaluation backend and is slower.

### GNN surrogate

```yaml
training:
  capacity_teacher:
    backend: gnn
    checkpoint: runs/capacity/geometry_gnn.pt
```

The checkpoint must be:

- `feature_mode = geometry_only`
- `schema_version = 2`
- `wrench_frame = contact_local`

The policy can exploit systematic surrogate errors, so final evaluation should always use the exact backend. High-scoring or uncertain policy-generated morphologies should also be re-labelled with exact statics and added to the capacity-GNN dataset.

## Actor

The canonical policy is one shared GRU (`architecture: gru`). No robot ID, global position, global graph, or policy-visible persistent heterogeneity is required. Every robot therefore executes the same local rule with its own observation history and GRU hidden state.

The legacy actor architectures remain in code for checkpoint compatibility/research probes, but the old YAML ablation suite was removed.

## Canonical run

```bash
python scripts/train.py \
  --config configs/capacity_local.yaml \
  --updates 2000
```

Checkpoint evaluation uses exact statics even if policy training used a GNN teacher:

```bash
python scripts/evaluate.py \
  runs/capacity_local/checkpoint_002000.pt \
  --config configs/capacity_local.yaml \
  --stage 4 --episodes 100
```

## Logged quantities

Episode logs include the ordinary simulator diagnostics plus:

- `reward_component_totals.capacity_terminal`
- `capacity_teacher.backend`
- `capacity_teacher.capacity_kg`
- `capacity_teacher.critical_component`
- `capacity_teacher.agent_credits`

These make it possible to distinguish global capacity improvement from local shaping rewards and inspect which robots received structural credit.

## How to tell whether the local policy actually learned

Do not select a policy from training reward alone. Use three separate checks.

1. **Training diagnostics.** `updates.jsonl` records PPO entropy, approximate KL,
   clip fraction, critic loss, and critic explained variance. `episodes.jsonl`
   records teacher capacity, terminal capacity reward, energy, anchors, falls,
   action distribution, and success. Plot them with:

   ```bash
   python scripts/plot_training.py runs/capacity_local --window 25
   ```

2. **Held-out exact evaluation across checkpoints.** Evaluate all saved policies
   on the same seeds that were not used as the training seed stream. Capacity is
   always recomputed with the exact 6-DOF statics teacher:

   ```bash
   python scripts/evaluate_checkpoints.py runs/capacity_local \
     --stage 4 --episodes 20 --seed-start 20000
   ```

   Choose the checkpoint with the best held-out exact capacity rather than
   automatically using the last checkpoint.

3. **Paired random baseline.** A trained policy should beat uniformly sampled
   valid actions on identical environments:

   ```bash
   python scripts/benchmark_policy.py \
     runs/capacity_local/checkpoint_002000.pt \
     --stage 4 --episodes 100 --seed-start 10000 \
     --output-dir runs/capacity_local/benchmark
   ```

   The benchmark reports mean/median capacity, success rate, paired capacity
   gain, the fraction of seeds on which the trained policy wins, and a bootstrap
   95% confidence interval for the mean paired capacity gain.

A useful policy should show positive held-out exact-capacity gain over random,
reasonable success at the target curriculum stage, and repeatable performance
across seeds. Entropy should normally decrease from its initial exploratory
level without collapsing immediately; very high clip fraction or rapidly
increasing approximate KL suggests PPO updates are too aggressive. Critic
explained variance moving toward positive values indicates that the centralized
critic is learning to predict returns, but it is diagnostic rather than a final
performance metric.

## Visualizing an actual learned rule

Render a deterministic held-out rollout:

```bash
python scripts/visualize_policy.py \
  runs/capacity_local/checkpoint_002000.pt \
  --stage 4 --seed 10000 \
  --output-prefix runs/capacity_local/episode_10000
```

This writes:

- `episode_10000.gif`: the construction process with current action counts;
- `episode_10000_final.png`: the final morphology with exact connector
  utilization and the critical contact emphasized;
- `episode_10000.json`: exact capacity, critical contact/component, energy,
  anchor/fall counts, and action distribution.

In the final utilization plot, connector color/width increases with

\[
\max_k |w_k|/w_{k,\max}.
\]

At the exact maximum payload the critical connector should be approximately at
utilization 1. This visualization is useful for spotting degenerate learned
behaviors such as all robots choosing the same motion, excessive anchoring,
unused branches, or a policy that succeeds only for one particular gap layout.
