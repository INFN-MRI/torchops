"""Unit tests for Linear Operator algebra."""

import pytest
import torch

from torchops import LinearOperator

# Define dummy functions for A and A_adjoint
def A(x):
    return 2 * x

def A_adj(x):
    return 2 * x

@pytest.fixture
def x():
    return torch.tensor([1.0])

@pytest.fixture
def F():
    return LinearOperator(A, A_adj)

@pytest.fixture
def G():
    return LinearOperator(A, A_adj)

# Test Add Operation
def test_add_operation(F, G, x):
    H = F + G
    assert isinstance(H, LinearOperator)
    assert H(x) == 4  # 2x + 2x = 4x

# Test Sub Operation
def test_sub_operation(F, G, x):
    H = F - G
    assert isinstance(H, LinearOperator)
    assert H(x) == 0  # 2x - 2x = 0

# Test Mul Operation
def test_mul_operation(F, x):
    G = 3 * F
    assert isinstance(G, LinearOperator)
    assert G(x) == 6  # 3 * (2x) = 6x

# Test Div Operation
def test_div_operation(F, x):
    G = F / 2
    assert isinstance(G, LinearOperator)
    assert G(x) == 1  # (2x) / 2 = x

# Test Neg Operation
def test_neg_operation(F, x):
    G = -F
    assert isinstance(G, LinearOperator)
    assert G(x) == -2  # - (2x) = -2x

# Test Matmul Operation
def test_matmul_operation(F, G, x):
    H = F @ G
    assert isinstance(H, LinearOperator)
    assert H(x) == 4  # (2x) @ (2x) = 4x
    assert H @ x == 4  # (2x) @ (2x) = 4x
    
# Test Power Operation
def test_pow_operation(F, G, x):
    FF = F**2
    assert isinstance(FF, LinearOperator)
    assert FF(x) == 4  # (2x) @ (2x) = 4x

# Test Adjoint Property
def test_adjoint_property(F, x):
    FH = F.adj
    assert isinstance(FH, LinearOperator)
    assert FH(x) == 2  # The adjoint should have A_adj and A swapped
    FH = F.adjoint
    assert isinstance(FH, LinearOperator)
    assert FH(x) == 2  # The adjoint should have A_adj and A swapped
    FH = F.H
    assert isinstance(FH, LinearOperator)
    assert FH(x) == 2  # The adjoint should have A_adj and A swapped

# Test Gram Property
def test_gram_property(F, x):
    FHF = F.gram
    assert isinstance(FHF, LinearOperator)
    assert FHF(x) == 4  # (2x) @ (2x) = 4x; gram = 4x
    FHF = F.N
    assert isinstance(FHF, LinearOperator)
    assert FHF(x) == 4  # (2x) @ (2x) = 4x; gram = 4x

# Test CompositeOperation Adjoint
def test_composite_operation_adjoint(F, G, x):
    H = F @ G
    HH = H.adjoint
    assert isinstance(HH, LinearOperator)



