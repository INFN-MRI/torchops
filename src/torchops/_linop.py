"""LinearOperator class."""

__all__ = ["LinearOperator"]

from deepinv.physics import LinearPhysics
from . import _mixin

class LinearOperator(_mixin.OperationMixin, LinearPhysics):
    def __init__(self, A, A_adjoint, max_iter=50, tol=1e-3):
        super().__init__(A=A, A_adjoint=A_adjoint, max_iter=max_iter, tol=tol)

    def _create_instance(self, operation):
        if isinstance(operation, _mixin.AddOperation):
            return AddOperation(operation.F, operation.G)
        elif isinstance(operation, _mixin.SubOperation):
            return SubOperation(operation.F, operation.G)
        elif isinstance(operation, _mixin.MulOperation):
            return MulOperation(operation.F, operation.a)
        elif isinstance(operation, _mixin.DivOperation):
            return DivOperation(operation.F, operation.a)
        elif isinstance(operation, _mixin.NegOperation):
            return NegOperation(operation.F)
        elif isinstance(operation, _mixin.CompositeOperation):
            return CompositeOperation(*operation.ops)
        else:
            raise ValueError("Unknown operation type")
            
    def _adjoint_op(self):
        return self.__class__(self.A_adj, self.A, self.max_iter, self.tol)

    @property
    def adj(self):
        return self._adjoint_op()

    @property
    def adjoint(self):
        return self._adjoint_op()

    @property
    def H(self):
        return self._adjoint_op()

    def _normal_op(self):
        return self.H @ self
    
    @property
    def gram(self):
        return self._normal_op()
    
    @property
    def N(self):
        return self._normal_op()

# %% Base operations for Linear Operators
class AddOperation(LinearOperator):
    def __init__(self, F, G):
        
        def A(x):
            return F.A(x) + G.A(x)
        
        def A_adjoint(x):
            return F.A_adjoint(x) + G.A_adjoint(x)
        
        super().__init__(A, A_adjoint, F.max_iter, F.tol)
        
        self.F = F
        self.G = G
        
    def _adjoint_op(self):
        return AddOperation(self.F.H, self.G.H)
        
class SubOperation(LinearOperator):
    def __init__(self, F, G):
        
        def A(x):
            return F.A(x) - G.A(x)
        
        def A_adjoint(x):
            return F.A_adjoint(x) - G.A_adjoint(x)
        
        super().__init__(A, A_adjoint, F.max_iter, F.tol)
        
        self.F = F
        self.G = G
    
    def _adjoint_op(self):
        return SubOperation(self.F.H, self.G.H)
        
class MulOperation(LinearOperator):
    def __init__(self, F, a):
        
        def A(x):
            return F.A(a * x)
        
        def A_adjoint(x):
            return F.A_adjoint(a * x)
        
        super().__init__(A, A_adjoint, F.max_iter, F.tol)
        
        self.F = F
        self.a = a
        
    def _adjoint_op(self):
        return MulOperation(self.F.H, self.a)
        
class DivOperation(LinearOperator):
    def __init__(self, F, a):
        
        def A(x):
            return F.A(x / a)
        
        def A_adjoint(x):
            return F.A_adjoint(x / a)
        
        super().__init__(A, A_adjoint, F.max_iter, F.tol)
        
        self.F = F
        self.a = a
        
    def _adjoint_op(self):
        return DivOperation(self.F.H, self.a)

class NegOperation(LinearOperator):
    def __init__(self, F):
        
        def A(x):
            return -F.A(x)
        
        def A_adjoint(x):
            return -F.A_adjoint(x)
        
        super().__init__(A, A_adjoint, F.max_iter, F.tol)
        
        self.F = F
        
    def _adjoint_op(self):
        return DivOperation(self.F.H)

class CompositeOperation(LinearOperator):
    def __init__(self, *ops):
        
        def A(x):
            y = x
            for op in reversed(ops):
                y = op(y)
            return y
        
        def A_adjoint(x):
            y = x
            for op in ops:
                y = op.H(y)
            return y
        
        super().__init__(A, A_adjoint, ops[0].max_iter, ops[0].tol)
        
        self.ops = ops
        
    def _adjoint_op(self):
        return CompositeOperation(*[op.H for op in reversed(self.ops)])
