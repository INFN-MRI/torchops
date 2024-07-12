"""Unit tests for Operator algebra."""

from torch import nn

from torchops._mixin import OperationMixin


class DummyOperator(OperationMixin, nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value * x


def test_add_operation():
    A = DummyOperator(2)
    B = DummyOperator(3)
    C = A + B
    assert isinstance(C, OperationMixin)


def test_sub_operation():
    A = DummyOperator(5)
    B = DummyOperator(3)
    C = A - B
    assert isinstance(C, OperationMixin)


def test_mul_operation():
    A = DummyOperator(2)
    C = A * 3
    assert isinstance(C, OperationMixin)


def test_div_operation():
    A = DummyOperator(6)
    C = A / 3
    assert isinstance(C, OperationMixin)


def test_neg_operation():
    A = DummyOperator(3)
    C = -A
    assert isinstance(C, OperationMixin)


def test_matmul_operation():
    A = DummyOperator(2)
    B = DummyOperator(3)
    C = A @ B
    assert isinstance(C, OperationMixin)


def test_pow_operation():
    A = DummyOperator(2)
    C = A**3
    assert isinstance(C, OperationMixin)
