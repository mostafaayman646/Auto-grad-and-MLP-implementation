import math

class Value:
    def __init__(self,data,operation = '',_children = (),label = ''):
    
        self.data = data
        self._op = operation
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
    
    def __add__(self, other):
    
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,'+', (self, other))
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
    
        other = other if isinstance(other,Value) else Value(other)
        out = Value(other.data * self.data, '*', (self,other))
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other,(int,float)), "Only supporting int or float power"
        out = Value(self.data**other,f'pow {other}', (self,))
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def Tanh(self):
    
        X = self.data
        t = (math.exp(2*X) - 1) / (math.exp(2*X) + 1)
        out = Value(t, 'tanh', (self,))
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def Relu(self):
        out = Value(0 if self.data < 0 else self.data, 'ReLU', (self,))
    
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
    
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
    
        self.grad = 1
        for node in reversed(topo):
            node._backward()
    
    def __truediv__(self, other):
        return self * other **-1
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + other * -1
    
    def __rsub__(self, other):
        return self*-1 + other
    
    def __rmul__(self, other): #other * self
        return self * other
    
    def __repr__(self):
        return f"Data: {self.data}"