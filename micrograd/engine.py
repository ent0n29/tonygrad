import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op='', label=''): #children->empty tuple
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None #by default it does nothing
        self._prev = set(_children) #in the class it will be a set
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        # in add the gradient distrubution flows equally to its childs
        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward #function that propagates the gradient

        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        #in mul we use the chain rule of derivatives
        def _backward():
            self.grad += other.data * out.grad  #chain rule!
            other.grad += self.data * out.grad  #chain rule!
        out._backward = _backward

        return out

    #using tanh function to "squish" the output
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh') #just one child, self, a tuple
    
        def _backward():
            #derivative of tanh chain ruled to out.grad into self.grad
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad
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
