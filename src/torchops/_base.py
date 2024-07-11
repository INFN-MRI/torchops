"""Base operator module."""

__all__ = ["BaseOperator"]

import torch.nn as nn

from ._algebra import OperationMixin


class BaseOperator(nn.Module, OperationMixin):
    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclasses")
