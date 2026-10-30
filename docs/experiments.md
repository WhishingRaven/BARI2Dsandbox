# Capacity-policy experiments

The current experimental question is not a comparison of many hand-designed actor architectures. The canonical actor is fixed and the experiments should probe whether global structural competence emerges from local sensing.

Recommended experiment axes:

1. **Teacher fidelity**: exact 6-DOF capacity teacher vs a geometry-only CapacityGNN surrogate.
2. **Observation ablation**: remove local strain, IR history, or action history while keeping the same shared GRU.
3. **Scale generalization**: train on one robot-count/gap range and test on larger bridges.
4. **Topology generalization**: compare morphology families that are rare or absent during training.
5. **Credit assignment**: terminal capacity only vs terminal capacity plus leave-one-robot-out difference reward.
6. **Surrogate exploitation**: compare GNN-teacher reward with exact post-hoc capacity and feed large discrepancies back into capacity-GNN training.

Primary metrics should include exact final capacity in kg, span rate, unsafe surrogate overprediction, construction energy, number of robots used/anchored/fallen, and the distribution of counterfactual structural credits.

Canonical training:

```bash
python scripts/train.py --config configs/capacity_local.yaml --updates 2000
```

Exact evaluation:

```bash
python scripts/evaluate.py \
  runs/capacity_local/checkpoint_002000.pt \
  --config configs/capacity_local.yaml \
  --stage 4 --episodes 100
```
