# 3-D / 6-DOF bridge statics

`bari2d.mechanics.spatial_statics` evaluates the wrench that the current BARI2D
contact graph would have to transmit under gravity and arbitrary external loads.
It is deliberately separate from the fast graph-flow load evaluator used by RL.

## Geometry adapter

The simulator stores `(x, y, theta, layer)`, not a full 3-D pose. The mechanics
adapter therefore uses:

- box length = `robot.length`
- box width = `robot.width`
- box height = `robot.climb_height`
- COM height = `(layer + 0.5) * climb_height`
- initial roll/pitch = 0
- initial yaw = `theta`

The static solve itself has six incremental DOFs per robot:

`(ux, uy, uz, rx, ry, rz)`.

For adjacent layers, the contact point is the centroid of the overlap patch of
the two projected rectangles at the horizontal layer interface. For same-layer
robots, it is the midpoint of the opposing lateral face centers. A bank anchor
acts at the bank-facing side-center; ordinary bank support acts at the bottom
face center.

## World and connector-local wrench

The solver still assembles equilibrium in world coordinates. At contact `c`, the
world-frame wrench on the edge source is

`w_world = (Fx, Fy, Fz, Mx, My, Mz)`.

Every contact also has a right-handed local frame `(n, t_u, t_v)`:

- `n`: contact normal, from `edge.source` toward `edge.target`;
- `t_u`, `t_v`: tangent axes in the contact plane.

The same physical wrench is resolved in that frame as

`w_local = (Axial, ShearU, ShearV, Torsion, BendingU, BendingV)`.

The component mapping is:

- `Axial = F . n`
- `ShearU = F . t_u`
- `ShearV = F . t_v`
- `Torsion = M . n`
- `BendingU = M . t_u`
- `BendingV = M . t_v`

This local representation is the preferred one for connector strength, because
a real anchor/connector normally has allowable axial, shear, bending and
torsional loads that do not depend on the global simulation axes.

For each rigid body, the assembled linear static equations enforce

`sum(F_c) + F_external = 0`

and

`sum((r_c x F_c) + M_c) + M_external = 0`.

Every existing contact graph edge is modeled as a welded 6-component linear
spring. This matches the current simulation rule that a contact does not detach
because its wrench is large. Strength limits are checked separately.

## Self weight and payload capacity

`SpatialStaticSolver.capacity()` keeps the legacy world-axis component limits.
For physical connector limits, use `SpatialStaticSolver.capacity_local()` or
`BridgeEnv.local_spatial_payload_capacity()`.

Both include robot self weight by default. Only the robot-to-robot connected
structure containing the payload is included; fixed bank nodes are terminal
supports, so unrelated robots standing on the same bank do not become part of
the bridge.

For a one-kilogram payload pattern the contact wrench is linear:

`w_c(m) = w_c,self + m * w_c,1kg`.

Therefore the maximum payload is obtained analytically from all six component
limits; no incremental mass sweep is required.

```python
from bari2d.mechanics import (
    LocalWrenchLimit,
    bridge_center_payload_3d,
)

payload = bridge_center_payload_3d(
    env.robots,
    env.graph,
    env.field,
    env.config.robot,
)

result = env.local_spatial_payload_capacity(
    payload,
    LocalWrenchLimit(
        axial=200.0,
        shear_u=120.0,
        shear_v=120.0,
        torsion=40.0,
        bending_u=80.0,
        bending_v=80.0,
    ),
)

print(result.max_payload_kg)
print(result.critical_contact)
print(result.critical_component)
```

For raw contact wrenches:

```python
response = env.spatial_static_response([payload])
for contact in response.contacts:
    print(contact.source, contact.target, contact.point)
    print("world:", contact.wrench_on_source)
    print("local:", contact.local_wrench_on_source)
    print("frame:", contact.frame)
```

`contact.wrench_on_target` and `contact.local_wrench_on_target` return the
equal-and-opposite wrench on the other endpoint.

## Important model boundary

This is a 6-DOF *static mechanics layer on top of a 2.5-D morphology*. It does
not turn the RL environment into a full 3-D multibody simulator. Initial robot
roll/pitch, finite contact patch pressure, Coulomb slip, geometric nonlinearity,
contact opening, buckling, and large-deflection sag are not modeled yet.

For dataset generation under the stated "contacts remain attached until a
wrench limit is exceeded" assumption, the linear model is consistent and gives
deterministic force/moment labels. A later full 3-D environment can reuse the
same wrench/capacity interfaces while replacing the geometry and stiffness
assembly.
