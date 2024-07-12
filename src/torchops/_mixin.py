"""Operator algebra module."""

class OperationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self._create_instance(AddOperation(self, other))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self._create_instance(SubOperation(self, other))
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return self._create_instance(MulOperation(self, scalar))
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return self._create_instance(DivOperation(self, scalar))
        return NotImplemented

    def __neg__(self):
        return self._create_instance(NegOperation(self))

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            return self._create_instance(CompositeOperation(self, other))
        else:
            return self(other)
        return NotImplemented
    
    def __pow__(self, scalar):
        if isinstance(scalar, (int)):
            ops = [self for n in range(scalar)]
            return self._create_instance(CompositeOperation(*ops))
        return NotImplemented
    
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        return self @ other
            
    def to(self, *args, **kwargs):
        for prop in self.__dict__.keys():
            try:
                self.__dict__[prop] = self.__dict__[prop].to(*args, **kwargs)
            except Exception:
                pass
        return self

    def _create_instance(self, operation):
        if isinstance(operation, AddOperation):
            return AddOperation(operation.F, operation.G)
        elif isinstance(operation, SubOperation):
            return SubOperation(operation.F, operation.G)
        elif isinstance(operation, MulOperation):
            return MulOperation(operation.F, operation.a)
        elif isinstance(operation, DivOperation):
            return DivOperation(operation.F, operation.a)
        elif isinstance(operation, NegOperation):
            return NegOperation(operation.F)
        elif isinstance(operation, CompositeOperation):
            return CompositeOperation(*operation.ops)
        else:
            raise ValueError("Unknown operation type")
            
            
# %% Base operations
class AddOperation(OperationMixin):
    def __init__(self, F, G):
        super().__init__()
        self.F = F
        self.G = G

    def forward(self, x):
        return self.F(x) + self.G(x)

class SubOperation(OperationMixin):
    def __init__(self, F, G):
        super().__init__()
        self.F = F
        self.G = G

    def forward(self, x):
        return self.F(x) - self.G(x)

class MulOperation(OperationMixin):
    def __init__(self, F, a):
        super().__init__()
        self.F = F
        self.a = a

    def forward(self, x):
        return self.F(self.a * x)

class DivOperation(OperationMixin):
    def __init__(self, F, a):
        super().__init__()
        self.F = F
        self.a = a

    def forward(self, x):
        return self.F(x / self.a)

class NegOperation(OperationMixin):
    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, x):
        return -self.F(x)

class CompositeOperation(OperationMixin):
    def __init__(self, *ops):
        super().__init__()
        self.ops = ops

    def forward(self, x):
        output = x
        for op in reversed(self.ops):
            output = op(output)
        return output


