"""A tiny fallback for the one ``tsai`` class used by upstream UnCLe.

The class is registered only when ``tsai`` is not installed; it avoids making a
large optional fastai stack a prerequisite for a single benchmark adapter.
"""

from __future__ import annotations

import sys
import types


def install_tcn_fallback() -> None:
    """Provide ``tsai.models.TCN.TemporalConvNet`` if upstream tsai is absent."""

    try:
        from tsai.models.TCN import TemporalConvNet  # noqa: F401

        return
    except Exception:
        pass

    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm

    class Chomp1d(nn.Module):
        def __init__(self, amount: int):
            super().__init__()
            self.amount = int(amount)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values if self.amount <= 0 else values[:, :, :-self.amount].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.network = nn.Sequential(
                weight_norm(nn.Conv1d(inputs, outputs, kernel_size, padding=padding, dilation=dilation)),
                Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                weight_norm(nn.Conv1d(outputs, outputs, kernel_size, padding=padding, dilation=dilation)),
                Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            )
            self.residual = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()
            self.relu = nn.ReLU()

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.relu(self.network(values) + self.residual(values))

    class TemporalConvNet(nn.Module):
        def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 2, dropout: float = 0.2, **_kwargs):
            super().__init__()
            layers = []
            for index, outputs in enumerate(num_channels):
                inputs = num_inputs if index == 0 else num_channels[index - 1]
                layers.append(TemporalBlock(inputs, outputs, kernel_size, 2**index, dropout))
            self.network = nn.Sequential(*layers)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.network(values)

    tsai = types.ModuleType("tsai")
    models = types.ModuleType("tsai.models")
    module = types.ModuleType("tsai.models.TCN")
    module.TemporalConvNet = TemporalConvNet
    models.TCN = module
    tsai.models = models
    sys.modules.setdefault("tsai", tsai)
    sys.modules.setdefault("tsai.models", models)
    sys.modules.setdefault("tsai.models.TCN", module)
