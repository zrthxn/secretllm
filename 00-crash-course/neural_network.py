import random
import math
import numpy as np

class Value:
    """
    A class to represent a value in a computational graph, supporting basic arithmetic operations and automatic differentiation.
    Attributes:
    -----------
    data : float
        The numerical value.
    grad : float
        The gradient of the value.
    _prev : set
        The set of parent nodes in the computational graph.
    _op : str
        The operation that produced this value.
    label : str
        An optional label for the value.
    _backward : function
        The function to compute the gradient of this value.
    Methods:
    --------
    __init__(data, _children=(), _op='', label=''):
        Initializes the Value object with data, children, operation, and label.
    _backward_placeholder():
        A placeholder backward function.
    __repr__():
        Returns a string representation of the Value object.
    __add__(other):
        Adds two Value objects or a Value object and a number.
    __radd__(other):
        Adds a number and a Value object.
    __neg__():
        Negates the Value object.
    __sub__(other):
        Subtracts a Value object or a number from the Value object.
    __rsub__(other):
        Subtracts the Value object from a number.
    __mul__(other):
        Multiplies two Value objects or a Value object and a number.
    __rmul__(other):
        Multiplies a number and a Value object.
    __truediv__(other):
        Divides the Value object by another Value object or a number.
    __rtruediv__(other):
        Divides a number by the Value object.
    __pow__(other):
        Raises the Value object to the power of an integer or float.
    exp():
        Computes the exponential of the Value object.
    relu():
        Applies the ReLU activation function to the Value object.
    tanh():
        Applies the tanh activation function to the Value object.
    log():
        Computes the natural logarithm of the Value object.
    __getstate__():
        Prepares the Value object for pickling.
    __setstate__(state):
        Restores the Value object from a pickled state.
    backward():
        Computes the gradients of all values in the computational graph using backpropagation.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = self._backward_placeholder

    def _backward_placeholder(self):
        pass
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return Value(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)  

    def __truediv__(self, other): # self / other
        return self * other ** -1 # (self) * (other^-1) precendence is set automatically
    
    def __rtruediv__(self, other): # other / self
        return Value(other) * self ** -1


    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
            # other.grad += self.data ** other * math.log(self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_backward'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._backward = self._backward_placeholder

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

class Neuron:
    """
    A class representing a single artificial neuron.
    Attributes:
    -----------
    w : list of Value
        Weights associated with the neuron's inputs.
    b : Value
        Bias term for the neuron.
    Methods:
    --------
    __init__(nin):
        Initializes the neuron with random weights and bias.
    __call__(x):
        Computes the output of the neuron for a given input `x` using a ReLU activation function.
    parameters():
        Returns the list of parameters (weights and bias) of the neuron.
    """

    def __init__(self, nin):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        self.b = Value(np.random.randn())
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    """
    Represents a layer in a neural network, consisting of multiple neurons.
    Attributes:
        neurons (list): A list of Neuron objects in the layer.
    Methods:
        __init__(nin, nout):
            Initializes the Layer with a specified number of input and output neurons.
        __call__(x):
            Applies the layer to an input x, returning the output of each neuron in the layer.
        parameters():
            Returns a list of all parameters from all neurons in the layer.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """
    Multi-Layer Perceptron (MLP) class for constructing and managing a neural network.
    Attributes:
        layers (list): A list of Layer objects representing the layers of the neural network.
    Methods:
        __init__(nin, nouts):
            Initializes the MLP with the given input size and list of output sizes for each layer.
        __call__(x):
            Passes the input through the network, layer by layer, and returns the output.
        parameters():
            Returns a list of all parameters in the network.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
