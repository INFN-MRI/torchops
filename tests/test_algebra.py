"""Unit tests for Operator algebra."""

import pytest
import torch

from torchops import BaseOperator


class ExampleOperation(BaseOperator):
    def forward(self, x):
        return x + 1  # Simple example implementation


@pytest.fixture
def tensor():
    return torch.tensor([5.0])


@pytest.fixture
def A():
    return ExampleOperation()


@pytest.fixture
def B():
    return ExampleOperation()


def test_addition(A, B, tensor):
    C = A + B
    result = C(tensor)
    expected = (tensor + 1) + (tensor + 1)
    assert torch.equal(result, expected)


def test_subtraction(A, B, tensor):
    D = A - B
    result = D(tensor)
    expected = (tensor + 1) - (tensor + 1)
    assert torch.equal(result, expected)


def test_multiplication(A, tensor):
    E = 2 * A
    result = E(tensor)
    expected = 2 * (tensor + 1)
    assert torch.equal(result, expected)


def test_division(A, tensor):
    F = A / 2
    result = F(tensor)
    expected = (tensor + 1) / 2
    assert torch.equal(result, expected)


def test_negation(A, tensor):
    G = -A
    result = G(tensor)
    expected = -(tensor + 1)
    assert torch.equal(result, expected)


def test_composite(A, B, tensor):
    H = A @ B
    result = H(tensor)
    expected = (tensor + 1) + 1
    assert torch.equal(result, expected)


# def test_conjugate(A, tensor):
#     I = A.conjugate()
#     result = I(tensor)
#     expected = torch.conj(tensor + 1)
#     assert torch.equal(result, expected)

# def test_transpose(A, tensor):
#     J = A.transpose()
#     result = J(tensor)
#     expected = (tensor + 1).transpose(0, 0)  # Transpose has no effect on 1D tensor
#     assert torch.equal(result, expected)

# def test_adjoint(A, tensor):
#     K = A.adjoint()
#     result = K(tensor)
#     expected = torch.conj((tensor + 1)).transpose(0, 0)  # Adjoint has no effect on 1D tensor
#     assert torch.equal(result, expected)

# def test_gram(A, tensor):
#     L = A.gram()
#     result = L(tensor)
#     adjoint_A = A.adjoint()
#     expected = (adjoint_A @ A)(tensor)
#     assert torch.equal(result, expected)
