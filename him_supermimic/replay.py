"""Time-major circular storage for vectorized physics transitions."""

import torch


class EpisodeReplay:
    def __init__(self, capacity: int, num_envs: int, *, fields=None):
        if min(capacity, num_envs) < 1:
            raise ValueError("capacity and num_envs must be positive")
        self.fields = fields
        self.capacity = capacity
        self.num_envs = num_envs
        self.total_steps = 0
        self.data: dict[str, torch.Tensor] = {}

    @property
    def size(self) -> int:
        return min(self.capacity, self.total_steps)

    @torch.inference_mode(False)
    @torch.no_grad()
    def append(self, transition: dict[str, torch.Tensor]) -> None:
        if self.fields is not None:
            transition = {name: transition[name] for name in self.fields}
        if any(value.shape[0] != self.num_envs for value in transition.values()):
            raise ValueError("Every field must have a row for each environment")
        if not self.data:
            self.data = {
                name: torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=value.device)
                for name, value in transition.items()
            }
        if self.data.keys() != transition.keys():
            raise ValueError("Transition fields changed")
        for name, value in transition.items():
            target = self.data[name]
            if (target.shape[1:], target.dtype, target.device) != (value.shape, value.dtype, value.device):
                raise ValueError(f"Transition field changed shape, dtype or device: {name}")
        index = self.total_steps % self.capacity
        for name, value in transition.items():
            self.data[name][index].copy_(value)
        self.total_steps += 1

    def ordered(self) -> dict[str, torch.Tensor]:
        if not self.data:
            raise ValueError("Replay is empty")
        device = self.data["episode_id"].device
        steps = torch.arange(self.total_steps - self.size, self.total_steps, device=device)
        return {name: value[steps % self.capacity] for name, value in self.data.items()}

    def valid_starts(self, horizon: int, *, allow_command_resampling: bool = False) -> torch.Tensor:
        """Return [absolute step, env] pairs. A boundary may terminate the last transition."""
        if not 1 <= horizon <= self.size:
            raise ValueError("horizon must be within the retained replay length")
        device = self.data["episode_id"].device
        first = self.total_steps - self.size
        steps = torch.arange(first, self.total_steps - horizon + 1, device=device)
        slots = steps % self.capacity
        valid = torch.ones((len(steps), self.num_envs), dtype=torch.bool, device=device)
        episode = self.data["episode_id"][slots]
        for offset in range(horizon):
            index = (steps + offset) % self.capacity
            valid &= self.data["episode_id"][index] == episode
            if not allow_command_resampling:
                valid &= ~self.data["command_resampled"][index]
            if offset < horizon - 1:
                valid &= ~(self.data["terminated"][index] | self.data["truncated"][index])
        pairs = valid.nonzero()
        pairs[:, 0] += first
        return pairs

    def sample(self, batch_size: int, horizon: int, *, allow_command_resampling: bool = False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = self.valid_starts(horizon, allow_command_resampling=allow_command_resampling)
        if len(starts) == 0:
            raise ValueError("No eligible replay windows")
        selection = starts[torch.randint(len(starts), (batch_size,), device=starts.device)]
        steps, envs = selection.unbind(dim=1)
        slots = (steps[:, None] + torch.arange(horizon, device=starts.device)) % self.capacity
        batch = {key: value[slots, envs[:, None]] for key, value in self.data.items()}
        batch["states"] = torch.cat((batch["state"][:, :1], batch["next_state_pre_reset"]), dim=1)
        batch["start_step"] = steps
        batch["env_id"] = envs
        return batch
