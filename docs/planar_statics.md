# Planar rigid-body statics prototype

`bari2d.mechanics.PlanarStaticSolver` is an in-plane verification model for the
current top-view BARI2D geometry. Each robot is a rigid body with small
displacement DOFs `(ux, uy, theta)`. Each current contact-graph edge is a 3-DOF
linear spring at the center of the corresponding robot face. Bank nodes are
fixed.

For a contact point `c`, the solver returns the world-frame wrench

`w = (Fx, Fy, Mz)`

where `Mz` is the pure couple transmitted at the contact. The moment of the
contact force about a robot center is computed separately as `r x F`, so robot
moment equilibrium uses `r x F + Mz`.

For a payload mass `m`, linearity gives

`w_c(m) = w_c,base + m * w_c,1kg`.

Therefore `PlanarStaticSolver.capacity()` computes the first component-wise
contact limit directly; it does not increase payload mass in discrete steps.
Different contacts can have different `PlanarWrenchLimit` values by passing a
mapping keyed by `frozenset((source, target))`.

Example:

```python
from bari2d.mechanics import PlanarWrenchLimit, bridge_center_payload

payload = bridge_center_payload(env.robots, env.graph, env.field)
result = env.planar_payload_capacity(
    payload,
    PlanarWrenchLimit(fx=100.0, fy=100.0, mz=30.0),
)
print(result.max_payload_kg)
print(result.critical_contact, result.critical_component)
```

This is not yet a gravity/sagging model. The simulator's `x-y` geometry is a
plan view and has no vertical translational DOF. The intended next step is a 3-D
rigid-body formulation with 6 DOFs per robot and a 6-component contact wrench
`(Fx, Fy, Fz, Mx, My, Mz)`. The present matrix assembly is deliberately written
so that extension is direct.
