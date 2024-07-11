"""Operator algebra module."""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class OperationMixin(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __add__(self, other):
        return AddOperation(self, other)

    def __sub__(self, other):
        return SubOperation(self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MulOperation(self, other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return DivOperation(self, other)
        else:
            return NotImplemented

    def __neg__(self):
        return NegOperation(self)

    def __matmul__(self, other):
        if isinstance(other, OperationMixin):
            return CompositeOperation(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        if isinstance(other, OperationMixin):
            return CompositeOperation(other, self)
        else:
            return NotImplemented

    # def conjugate(self):
    #     return ConjugateOperation(self)

    # def transpose(self):
    #     return TransposeOperation(self)

    # def adjoint(self):
    #     return AdjointOperation(self)

    # def gram(self):
    #     return GramOperation(self)


# Supporting Operation Classes
class AddOperation(OperationMixin, nn.Module):
    def __init__(self, A, B):
        super(AddOperation, self).__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A(x) + self.B(x)


class SubOperation(OperationMixin, nn.Module):
    def __init__(self, A, B):
        super(SubOperation, self).__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A(x) - self.B(x)


class MulOperation(OperationMixin, nn.Module):
    def __init__(self, A, c):
        super(MulOperation, self).__init__()
        self.A = A
        self.c = c

    def forward(self, x):
        return self.c * self.A(x)


class DivOperation(OperationMixin, nn.Module):
    def __init__(self, A, c):
        super(DivOperation, self).__init__()
        self.A = A
        self.c = c

    def forward(self, x):
        return self.A(x) / self.c


class NegOperation(OperationMixin, nn.Module):
    def __init__(self, A):
        super(NegOperation, self).__init__()
        self.A = A

    def forward(self, x):
        return -self.A(x)


class CompositeOperation(OperationMixin, nn.Module):
    def __init__(self, A, B):
        super(CompositeOperation, self).__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A(self.B(x))


# class ConjugateOperation(OperationMixin, nn.Module):
#     def __init__(self, A):
#         super(ConjugateOperation, self).__init__()
#         self.A = A

#     def forward(self, x):
#         return torch.conj(self.A(x))

# class TransposeOperation(OperationMixin, nn.Module):
#     def __init__(self, A):
#         super(TransposeOperation, self).__init__()
#         self.A = A

#     def forward(self, x):
#         return self.A(x).transpose()

# class AdjointOperation(OperationMixin, nn.Module):
#     def __init__(self, A):
#         super(AdjointOperation, self).__init__()
#         self.A = A

#     def forward(self, x):
#         return torch.conj(self.A(x)).transpose()

# class GramOperation(OperationMixin, nn.Module):
#     def __init__(self, A):
#         super(GramOperation, self).__init__()
#         self.A = A

#     def forward(self, x):
#         return (self.A.adjoint() @ self.A)(x)
