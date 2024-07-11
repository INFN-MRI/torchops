# TorchOps

`OperationMixin` is a Python library that extends the functionality of PyTorch's `nn.Module` with additional mathematical operations and transformations. This library provides an abstract base class that supports various linear algebra operations such as addition, subtraction, multiplication, division, and more. It is designed for use in machine learning and scientific computing tasks where such operations are frequently needed.

## Features

- **Basic Operations**: Addition, Subtraction, Multiplication, Division, Negation
- **Composite Operations**: Composition of operations with `@` (matrix multiplication-like)
- **Advanced Operations**: Conjugate, Transpose, Adjoint, Gram Matrix
- **Extensibility**: Easy to extend with new operators and transformations

## Installation

You can install `TorchOps` via pip. If you are not using a virtual environment, you might need to use `pip install --user` to avoid permission issues.

```bash
pip install git+https://github.com/INFN-MRI/torchops.git@main
```

## Basic Usage

Here's a quick example of how to use `TorchOps` with custom PyTorch modules:

### Define a Custom Module

```python

import torch
import torch.nn as nn

from torchops import BaseOperator

class CustomOperator(BaseOperator):
    def forward(self, x):
        return x + 1  # Simple example implementation
```

### Using the Operations

```python
A = CustomOperator()
B = CustomOperator()
x = torch.tensor([5.0])

# Addition
C = A + B
print(C(x))  # Output: tensor([12.0])

# Subtraction
D = A - B
print(D(x))  # Output: tensor([0.0])

# Multiplication
E = 2 * A
print(E(x))  # Output: tensor([12.0])

# Division
F = A / 2
print(F(x))  # Output: tensor([3.0])

# Negation
G = -A
print(G(x))  # Output: tensor([-6.0])

# Composite
H = A @ B
print(H(x))  # Output: tensor([7.0])

# Conjugate
I = A.conjugate()
print(I(x))  # Output: tensor([6.0])

# Transpose
J = A.transpose()
print(J(x))  # Output: tensor([6.0])

# Adjoint
K = A.adjoint()
print(K(x))  # Output: tensor([6.0])

# Gram
L = A.gram()
print(L(x))  # Output: depends on implementation of adjoint and @
```

## Testing

To ensure that everything is working correctly, run the test suite using `pytest`:

```bash
pytest .
```

## Contributing

We welcome contributions from the community! To contribute, please follow these steps:

1. **Fork the Repository**: Create a fork of this repository on GitHub.
2. **Create a Branch**: Create a new branch for your changes.
3. **Make Changes**: Implement your changes or features.
4. **Submit a Pull Request**: Open a pull request with a description of your changes.

Please make sure to include tests for any new features or bug fixes and follow the existing code style guidelines.

## Resources

- [PyTorch](https://pytorch.org/): The deep learning framework used by `OperationMixin`.
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/): Documentation generator used for `OperationMixin`.
- [MyST-Parser](https://myst-parser.readthedocs.io/en/latest/): Markdown parser for Sphinx.

## License

`TorchOps` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Thanks to the PyTorch community for providing a robust and flexible machine learning framework.
- Thanks to the open-source community for their contributions to the tools and libraries used in this project.

## Contact

For any questions or feedback, please open an issue on GitHub or contact us.
