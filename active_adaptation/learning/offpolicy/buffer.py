import torch
from tensordict import TensorDict
from typing import Optional, Tuple


class ReplayBuffer:
    def __init__(self, max_size: int, fake_tensordict: TensorDict):
        self.max_size = max_size
        self.num_envs = fake_tensordict.shape[0]
        self.device = fake_tensordict.device
        self._current_size = 0
        self._td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        self._ptr = 0

    def push(self, tensordict: TensorDict):
        self._td[self._ptr] = tensordict.to(self.device)
        self._ptr = (self._ptr + 1) % self._td.shape[0]
        self._current_size = min(self._current_size + 1, self.max_size)
    
    append = push
    
    def last(self, steps: int) -> TensorDict:
        """
        Returns the last `steps` samples from the buffer.
        """
        assert len(self) > steps, "Not enough samples in buffer"
        if self._ptr > steps:
            samples = self._td[self._ptr - steps:self._ptr].clone()
        else:
            part1 = self._td[-(steps - self._ptr):]
            part2 = self._td[:self._ptr]
            samples = torch.cat([part1, part2], dim=0)
        assert samples.shape[0] == steps, "Not enough samples in buffer"
        return samples

    @property
    def num_samples(self):
        return self._td.shape[1] * len(self)

    def sample(self, batch_size: int, steps: int=1) -> TensorDict:
        if steps == 1:
            indices = torch.randint(0, self.num_samples, (batch_size,), device=self._td.device)
            samples = self._td.view(-1)[indices]
        else:
            indices = torch.randint(0, self.num_samples, (batch_size,), device=self._td.device)
            t, e = torch.unravel_index(indices, (len(self), self._td.shape[1]))
            t = (t.unsqueeze(0) + torch.arange(steps, device=self._td.device).unsqueeze(1)) % len(self)
            samples = self._td[t, e]
            assert samples.shape[:2] == (steps, batch_size)
        return samples
    
    def sample_sequential(
        self,
        batch_size: int,
        steps: int=1,
        last_indices: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        sequential_prob: float=0.0,
        sequential_offset: int=-1,
    ) -> Tuple[TensorDict, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample transitions with optional temporal correlation along the replay ring.

        Most indices are drawn i.i.d., like :meth:`sample`. With probability
        ``sequential_prob``, each element instead steps **backward** along the ring
        time index for the same env id, matching the direction rewards propagate
        under dynamic programming (earlier stored step = one step back in ring).

        Ring indices wrap with ``% len(self)``. After a minibatch, pass the returned
        ``(t, e)`` back as ``last_indices`` on the next call to chain segments.

        Args:
            batch_size: Number of transitions (or sequences when ``steps > 1``).
            steps: If ``> 1``, return length-``steps`` segments starting at each
                chosen ``t`` (forward along the ring from that start).
            last_indices: Previous ``(t, e)`` tensors from this method, same length
                as ``batch_size``, or ``None`` for independent draws only.
            sequential_prob: In ``(0, 1]``, fraction of elements (in expectation)
                that reuse ``last_indices`` shifted by ``sequential_offset`` instead
                of a fresh random index. ``0`` disables chaining.
            sequential_offset: Added to ``last_indices[0]`` before wrapping; default
                ``-1`` is one step **backward** in ring time (reward backup direction).

        Returns:
            ``(samples, (t, e))`` — batch/sequence tensordict and the time/env indices
            used for each row (for feeding back as ``last_indices``).
        """
        # sample new indices
        indices_flat = torch.randint(0, self.num_samples, (batch_size,), device=self._td.device)
        t, e = torch.unravel_index(indices_flat, (len(self), self._td.shape[1]))
        
        if last_indices is not None and sequential_prob > 0.0:
            use_new = torch.rand(batch_size, device=self._td.device) > sequential_prob
            t = torch.where(use_new, t, (last_indices[0] + sequential_offset) % len(self))
            e = torch.where(use_new, e, last_indices[1])
        
        if steps == 1:
            samples = self._td[t, e].squeeze(0)
        else:
            ts = (t.unsqueeze(0) + torch.arange(steps, device=self._td.device).unsqueeze(1)) % len(self)
            samples = self._td[ts, e]
            assert samples.shape[:2] == (steps, batch_size)
        
        return samples, (t, e)

    def __len__(self):
        return self._current_size

