from __future__ import annotations

import pytest
import torch

from bari2d.models.actor import ARCHITECTURES, SharedRecurrentActor
from bari2d.models.critic import CentralizedCritic
from bari2d.models.graph_critic import GraphCritic
from bari2d.utils.config import ModelConfig


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_all_actor_ablations_produce_valid_shapes(env, architecture: str) -> None:
    config = ModelConfig(architecture=architecture, branch_hidden=16, fused_hidden=24, recurrent_hidden=20)
    actor = SharedRecurrentActor(env.layout, env.action_count, config)
    observations = torch.as_tensor(env.observations()[:3])
    hidden = actor.initial_hidden(3)
    mask = torch.as_tensor(env.action_masks()[:3])
    output = actor.forward_step(observations, hidden, mask)
    assert output.logits.shape == (3, env.action_count)
    assert output.hidden.shape == (3, 20)
    assert torch.isfinite(output.logits).all()


def test_actor_sequence_and_auxiliary_heads(env) -> None:
    config = ModelConfig(architecture="bio_film", branch_hidden=16, fused_hidden=24, recurrent_hidden=20)
    actor = SharedRecurrentActor(env.layout, env.action_count, config)
    observation = torch.as_tensor(env.observations()[:3])
    observations = observation.unsqueeze(0).repeat(4, 1, 1)
    actions = torch.zeros((4, 3), dtype=torch.long)
    masks = torch.ones((4, 3, env.action_count), dtype=torch.bool)
    resets = torch.zeros((4, 3))
    log_probabilities, entropies, auxiliary = actor.evaluate_sequence(
        observations, actor.initial_hidden(3), actions, masks, resets
    )
    assert log_probabilities.shape == (4, 3)
    assert entropies.shape == (4, 3)
    assert set(auxiliary) == {"traffic", "connectivity", "contact_persistence", "force_trend"}


def test_centralized_critics_have_scalar_values(env) -> None:
    mlp = CentralizedCritic(env.global_state_size)
    assert mlp(torch.as_tensor(env.global_state()).unsqueeze(0)).shape == (1,)
    nodes, adjacency, edges = env.dense_graph_state()
    graph = GraphCritic()
    assert graph(torch.as_tensor(nodes), torch.as_tensor(adjacency), torch.as_tensor(edges)).ndim == 0

