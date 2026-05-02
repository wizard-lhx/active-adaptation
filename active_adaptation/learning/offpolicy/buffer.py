import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, max_size: int, fake_tensordict: TensorDict):
        self.max_size = max_size
        self._current_size = 0
        self._td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        self._ptr = 0

    def push(self, tensordict: TensorDict):
        self._td[self._ptr] = tensordict
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

    def __len__(self):
        return self._current_size

