import torch
import torch.distributions as D
import math


class ScaledTanhNormal(D.Distribution):
    """
    Different from torchrl's TanhNormal which uses upscale as
    .. math::
        loc = tanh(loc / upscale) * upscale,
    this class uses
    .. math::
        sample = tanh(sample / upscale) * upscale.
    """
    def __init__(
        self,
        loc: torch.Tensor,
        std: torch.Tensor,
        upscale: torch.Tensor,
        event_dims: int = 1
    ):
        self.loc = loc
        self.std = std
        self.upscale = upscale
        self.device = loc.device
        self.event_dims = event_dims
    
    @property
    def batch_shape(self) -> torch.Size:
        return self.loc.shape[:-self.event_dims]

    @property
    def event_shape(self) -> torch.Size:
        return self.loc.shape[-self.event_dims:]

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        eps = torch.randn(*sample_shape, *self.std.shape, device=self.device)
        sample = self.loc + self.std * eps
        return torch.tanh(sample / self.upscale) * self.upscale

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Invert the squash: a = tanh(x/s)*s  =>  x = atanh(a/s)*s
        u = (value / self.upscale).clamp(-1 + 1e-6, 1 - 1e-6)
        x = torch.atanh(u) * self.upscale
        # Gaussian log-prob at the pre-tanh sample
        log_p = -0.5 * (
            ((x - self.loc) / self.std).pow(2)
            + 2 * self.std.log()
            + math.log(2 * math.pi)
        )
        # Jacobian: d(tanh(x/s)*s)/dx = 1 - tanh²(x/s) = 1 - (a/s)²
        log_det = torch.log(1 - u.pow(2) + 1e-6)
        return (log_p + log_det).sum(dim=-self.event_dims)