from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from bari2d.env.bridge_env import ObservationLayout
from bari2d.models.encoders import (
    ConnectivityEncoder,
    FiLMConditioner,
    GoalEncoder,
    MechanicalEncoder,
    TemporalEncoder,
    TrafficEncoder,
)
from bari2d.utils.config import ModelConfig


ARCHITECTURES = {
    "mlp",
    "gru",
    "gru_traffic",
    "gru_traffic_connectivity",
    "bio",
    "bio_film",
    "bio_heterogeneity",
    "structured_mappo",
}

CONNECTIVITY_INPUTS = {"ir", "ir_strain_action_cue"}


@dataclass
class ActorOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    auxiliary: dict[str, torch.Tensor]
    latents: dict[str, torch.Tensor]


class RecurrentCore(nn.GRU):
    """Single-layer GRU used step by step while acting and over whole sequences in training.

    Checkpoints written before the switch from ``nn.GRUCell`` store the same
    weights under ``weight_ih``/``weight_hh``/``bias_ih``/``bias_hh``; they are
    renamed on load (the gate layout of GRUCell and GRU is identical).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[override]
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            legacy, current = prefix + name, prefix + name + "_l0"
            if legacy in state_dict and current not in state_dict:
                state_dict[current] = state_dict.pop(legacy)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class SharedRecurrentActor(nn.Module):
    def __init__(self, layout: ObservationLayout, action_count: int, config: ModelConfig):
        super().__init__()
        if config.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown actor architecture: {config.architecture}")
        if config.connectivity_inputs not in CONNECTIVITY_INPUTS:
            raise ValueError(f"Unknown connectivity_inputs: {config.connectivity_inputs}")
        self.layout = layout
        self.action_count = action_count
        self.config = config
        self.is_structured = config.architecture == "structured_mappo"
        self.hidden_size = config.recurrent_hidden
        branch = config.branch_hidden
        self.cue_size = layout.cues.stop - layout.cues.start
        self.rich_connectivity = self.is_structured and config.connectivity_inputs == "ir_strain_action_cue"
        self.use_traffic = config.architecture in {
            "gru_traffic", "gru_traffic_connectivity", "bio", "bio_film", "bio_heterogeneity", "structured_mappo"
        }
        self.use_connectivity = config.architecture in {
            "gru_traffic_connectivity", "bio", "bio_film", "bio_heterogeneity", "structured_mappo"
        }
        self.use_mechanical = config.architecture in {"bio", "bio_film", "bio_heterogeneity", "structured_mappo"}
        self.use_film = config.film or config.architecture in {"bio_film", "structured_mappo"}
        self.use_heterogeneity = config.use_heterogeneity or config.architecture == "bio_heterogeneity"
        self.is_recurrent = config.architecture != "mlp"

        self.goal_encoder = GoalEncoder(branch)
        if self.use_traffic:
            self.traffic_encoder = TemporalEncoder(layout.ray_count, branch) if self.is_structured else TrafficEncoder(layout.ray_count, branch)
            self.traffic_aux = nn.Linear(branch, 1)
        if self.use_connectivity:
            if self.is_structured:
                channels = layout.ray_count
                if self.rich_connectivity:
                    channels += 1 + action_count + self.cue_size  # strain, previous action (one-hot), cues
                self.connectivity_encoder = TemporalEncoder(channels, branch)
            else:
                self.connectivity_encoder = ConnectivityEncoder(layout.ray_count, branch)
            self.connectivity_aux = nn.Linear(branch, 2)
        if self.use_mechanical:
            self.mechanical_encoder = TemporalEncoder(1, branch) if self.is_structured else MechanicalEncoder(branch)
            self.force_aux = nn.Linear(branch, 1)
        if self.use_film:
            if self.use_traffic:
                self.traffic_film = FiLMConditioner(branch, branch)
            if self.use_mechanical:
                self.mechanical_film = FiLMConditioner(branch, branch)

        if config.architecture in {"mlp", "gru"}:
            fused_input = layout.size
        else:
            fused_input = branch  # goal
            fused_input += branch if self.use_traffic else 0
            fused_input += branch if self.use_connectivity else 0
            fused_input += branch if self.use_mechanical else layout.history
            fused_input += 6
            fused_input += layout.beacon.stop - layout.beacon.start
            fused_input += self.cue_size
            if self.use_heterogeneity:
                fused_input += layout.latent.stop - layout.latent.start
        self.fused = nn.Sequential(nn.Linear(fused_input, config.fused_hidden), nn.Tanh())
        if self.is_recurrent:
            self.recurrent = RecurrentCore(config.fused_hidden, config.recurrent_hidden)
            policy_input = config.recurrent_hidden
        else:
            policy_input = config.fused_hidden
        self.policy_head = nn.Linear(policy_input, action_count)

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _split(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = observation.shape[0]
        return {
            "ir": observation[:, self.layout.ir].reshape(batch, self.layout.history, self.layout.ray_count),
            "strain": observation[:, self.layout.strain],
            "actions": observation[:, self.layout.action_history],
            "internal": observation[:, self.layout.internal],
            "beacon": observation[:, self.layout.beacon],
            "cues": observation[:, self.layout.cues],
            "goal": observation[:, self.layout.goal],
            "latent": observation[:, self.layout.latent],
        }

    def _connectivity_sequence(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-history-step input of the structured connectivity branch: (batch, history, channels)."""
        ir = parts["ir"]
        if not self.rich_connectivity:
            return ir
        batch, history = ir.shape[0], ir.shape[1]
        strain = parts["strain"].unsqueeze(-1)
        # The observation stores the previous actions as a / (A - 1); recover the index.
        indices = torch.round(parts["actions"] * (self.action_count - 1)).long().clamp(0, self.action_count - 1)
        actions = nn.functional.one_hot(indices, self.action_count).to(ir.dtype)
        cues = parts["cues"].unsqueeze(1).expand(batch, history, self.cue_size)
        return torch.cat((ir, strain, actions, cues), dim=-1)

    def encode(self, observation: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perception branches and fusion for a flat batch of observations (batch, obs).

        Every row is independent, so a whole rollout (T, N, D) is encoded at
        once as (T * N, D) before the recurrent core.
        """
        parts = self._split(observation)
        latents: dict[str, torch.Tensor] = {}
        auxiliary: dict[str, torch.Tensor] = {}
        if self.config.architecture in {"mlp", "gru"}:
            fused_values = observation
        else:
            goal = self.goal_encoder(parts["goal"])
            latents["goal"] = goal
            values: list[torch.Tensor] = []
            if self.use_traffic:
                traffic = self.traffic_encoder(parts["ir"]) if self.is_structured else self.traffic_encoder(parts["ir"], parts["actions"])
                if self.use_film:
                    traffic = self.traffic_film(traffic, goal)
                latents["traffic"] = traffic
                auxiliary["traffic"] = self.traffic_aux(traffic).squeeze(-1)
                values.append(traffic)
            if self.use_connectivity:
                connectivity = (
                    self.connectivity_encoder(self._connectivity_sequence(parts))
                    if self.is_structured
                    else self.connectivity_encoder(parts["ir"], parts["strain"], parts["actions"], parts["internal"][:, 2:3])
                )
                latents["connectivity"] = connectivity
                connectivity_output = self.connectivity_aux(connectivity)
                auxiliary["connectivity"] = connectivity_output[:, 0]
                auxiliary["contact_persistence"] = connectivity_output[:, 1]
                values.append(connectivity)
            if self.use_mechanical:
                mechanical = (
                    self.mechanical_encoder(parts["strain"].unsqueeze(-1))
                    if self.is_structured
                    else self.mechanical_encoder(parts["strain"], parts["actions"])
                )
                if self.use_film:
                    mechanical = self.mechanical_film(mechanical, goal)
                latents["mechanical"] = mechanical
                auxiliary["force_trend"] = self.force_aux(mechanical).squeeze(-1)
                values.append(mechanical)
            else:
                values.append(parts["strain"])
            values.extend((goal, parts["internal"]))
            if parts["beacon"].shape[-1]:
                values.append(parts["beacon"])
            if parts["cues"].shape[-1]:
                values.append(parts["cues"])
            if self.use_heterogeneity:
                values.append(parts["latent"])
            fused_values = torch.cat(values, dim=-1)
        return self.fused(fused_values), auxiliary, latents

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> ActorOutput:
        if reset_mask is not None:
            hidden = hidden * (1.0 - reset_mask.float().reshape(-1, 1))
        fused, auxiliary, latents = self.encode(observation)
        if self.is_recurrent:
            _, next_hidden = self.recurrent(fused.unsqueeze(0), hidden.unsqueeze(0).contiguous())
            next_hidden = next_hidden.squeeze(0)
            policy_features = next_hidden
        else:
            next_hidden = hidden
            policy_features = fused
        logits = self.policy_head(policy_features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        return ActorOutput(logits, next_hidden, auxiliary, latents)

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActorOutput]:
        output = self.forward_step(observation, hidden, action_mask, reset_mask)
        distribution = Categorical(logits=output.logits)
        action = output.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), output

    def _run_recurrent(self, fused: torch.Tensor, initial_hidden: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
        """Run the GRU over (T, B, F) inputs; a reset at step t restarts that column from zero.

        The whole sequence goes through one ``nn.GRU`` call.  Columns that
        restart inside the sequence (an episode ended there) are re-run from
        the restart step with a zero hidden state and patched in, in order of
        the restart step, so later restarts overwrite earlier ones.
        """
        steps, batch = fused.shape[0], fused.shape[1]
        start = initial_hidden * (~resets[0]).to(initial_hidden.dtype).unsqueeze(-1)
        output, _ = self.recurrent(fused, start.unsqueeze(0).contiguous())
        restart_steps = torch.nonzero(resets[1:].any(dim=1), as_tuple=False).flatten() + 1
        for step in restart_steps.tolist():
            columns = resets[step]
            rerun, _ = self.recurrent(fused[step:, columns], fused.new_zeros(1, int(columns.sum()), self.hidden_size))
            patched = torch.zeros_like(output)
            patched[step:, columns] = rerun
            region = torch.zeros(steps, batch, 1, dtype=torch.bool, device=fused.device)
            region[step:, columns] = True
            output = torch.where(region, patched, output)
        return output

    def sequence_logits(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        reset_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked logits (T, B, A) and auxiliary predictions (T, B) for a batch of sequences."""
        steps, batch = observations.shape[0], observations.shape[1]
        fused, auxiliary, _ = self.encode(observations.reshape(steps * batch, -1))
        fused = fused.reshape(steps, batch, -1)
        auxiliary = {name: value.reshape(steps, batch) for name, value in auxiliary.items()}
        if self.is_recurrent:
            resets = (
                torch.zeros(steps, batch, dtype=torch.bool, device=observations.device)
                if reset_masks is None
                else reset_masks > 0.5
            )
            features = self._run_recurrent(fused, initial_hidden, resets)
        else:
            features = fused
        logits = self.policy_head(features)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), -1.0e9)
        return logits, auxiliary

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        reset_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logits, auxiliary = self.sequence_logits(observations, initial_hidden, action_masks, reset_masks)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), auxiliary
