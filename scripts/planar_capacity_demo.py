from __future__ import annotations

import argparse

from bari2d.env.bridge_env import BridgeEnv
from bari2d.mechanics import PlanarWrenchLimit, bridge_center_payload
from bari2d.utils.scripted import assemble_parallel_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planar contact-wrench capacity solver")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--fx-limit", type=float, default=100.0)
    parser.add_argument("--fy-limit", type=float, default=100.0)
    parser.add_argument("--mz-limit", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    env = BridgeEnv()
    env.reset(seed=args.seed)
    assemble_parallel_bridge(env, rows=args.rows, anchor=True)

    payload = bridge_center_payload(env.robots, env.graph, env.field)
    limits = PlanarWrenchLimit(args.fx_limit, args.fy_limit, args.mz_limit)
    response = env.planar_static_response([payload])
    capacity = env.planar_payload_capacity(payload, limits)

    print(f"spans: {env.graph.spans}")
    print(f"payload robot: {payload.robot_id}")
    print(f"stable: {response.stable} (rank {response.rank}/{response.dof_count})")
    print(f"equilibrium residual: {response.residual_norm:.3e}")
    print("contact wrenches for 1 kg test payload:")
    for contact in response.contacts:
        w = contact.wrench_on_source
        if abs(w.fx) + abs(w.fy) + abs(w.mz) < 1.0e-10:
            continue
        print(
            f"  {contact.source!r} -> {contact.target!r} [{contact.kind}] "
            f"at ({contact.point[0]:.3f}, {contact.point[1]:.3f}): "
            f"Fx={w.fx:.4f} N, Fy={w.fy:.4f} N, Mz={w.mz:.4f} N m"
        )

    print(f"max payload: {capacity.max_payload_kg:.6g} kg")
    print(f"failure mode: {capacity.failure_mode}")
    print(f"critical contact: {capacity.critical_contact}")
    print(f"critical component: {capacity.critical_component}")


if __name__ == "__main__":
    main()
