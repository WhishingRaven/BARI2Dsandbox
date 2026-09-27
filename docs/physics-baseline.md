# Physics baseline

BARI_26 imports the physical environment from `qndus521-lab/BARI` commit `198a04b23aac03f2284aeeaeca346c014c7747d0` without modification.

The protected physics scope is the complete `bari2d/env/` package:

- `robot.py`: rectangular robot state, kinematics, climbing layer state
- `field.py`: randomized banks, gap geometry, and terrain queries
- `sensors.py`: planar and downward IR sensing
- `contact_model.py`: contact, friction, anchoring, breakage, and graph construction
- `load_evaluator.py`: fast and incremental structural load tests
- `bridge_env.py`: simulator lifecycle, local observations, reward, and terminal evaluation

The BARI_26 change set is confined to the actor architecture, training defaults, configurations, documentation, and tests. Before the initial release, SHA-256 hashes of every file in `bari2d/env/` were checked against the source snapshot and matched exactly.

## Opt-in static physics

`environment.physics.model` defaults to `legacy`, which runs the protected physics above unchanged: `contact_model.py`, `load_evaluator.py`, `robot.py`, `field.py` and `sensors.py` are untouched, and `bridge_env.py` only dispatches on the physics model. Setting it to `static` enables gravity and rigid-body equilibrium through the new modules `statics.py` and `static_physics.py` (see docs/simulation.md, section 9). Checkpoints saved before this option existed restore as `legacy`.
