import math

# Minimal Value class (like micrograd)
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Function to run 1 neuron forward + backward pass
def run_neuron(x_val):
    x = Value(x_val)
    w = Value(2.0)
    b = Value(0.0)
    
    y = (w * x + b).tanh()
    y.backward()
    
    print(f"\nInput x = {x_val}")
    print(f"Output y = tanh(w*x + b) = {y.data:.5f}")
    print(f"dy/dx = {x.grad:.5f}")
    print(f"dy/dw = {w.grad:.5f}")
    print(f"dy/db = {b.grad:.5f}")

# Case 1: Input is small (no saturation)
run_neuron(1.0)

# Case 2: Input is large (saturated tanh)
run_neuron(10.0)
