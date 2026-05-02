import torch
import torch.distributions as D
from torchrl.modules.distributions.continuous import (
    FasterTransformedDistribution,
    SafeTanhTransform
)


class ScaledTanhNormal(FasterTransformedDistribution):
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

        t = D.ComposeTransform([
            D.AffineTransform(loc=0.0, scale=1/self.upscale),
            SafeTanhTransform(),
            D.AffineTransform(loc=0.0, scale=self.upscale),
        ])
        base_dist = D.Independent(
            D.Normal(self.loc, self.std),
            reinterpreted_batch_ndims=event_dims,
        )
        super().__init__(base_dist, t)

