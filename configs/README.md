# Training configuration

The old actor/critic ablation YAMLs were removed. There is now one canonical policy-training setup:

- `capacity_local.yaml`

The research question is no longer which hand-designed actor variant wins. Training is organized as centralized training / decentralized execution (CTDE):

1. every robot runs the same recurrent actor and receives only its local observation/history;
2. the training-only graph critic may inspect the global construction state;
3. at the end of a fixed construction horizon, the capacity teacher scores the completed morphology;
4. that capacity becomes the global terminal return;
5. leave-one-robot-out capacity differences provide agent-specific structural credit;
6. neither the graph critic nor the capacity teacher is available to deployed robots.

Run:

```bash
python scripts/train.py --config configs/capacity_local.yaml --updates 2000
```

## Capacity teacher backends

`training.capacity_teacher.backend: exact` uses the 6-DOF statics solver and is the ground-truth default.

After a trustworthy `geometry_only` capacity GNN has been trained, switch to:

```yaml
training:
  capacity_teacher:
    backend: gnn
    checkpoint: runs/capacity/geometry_gnn.pt
```

The GNN checkpoint must use schema version 2, `wrench_frame=contact_local`, and `feature_mode=geometry_only`. Exact statics should still be used for validation because a policy can exploit surrogate-model errors.
