# tonygrad
 
 A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.

### Installation

```bash
pip install tonygrad
```

### Example usage

Below there is a simple example showing a number of possible supported operations:

```python
"""tanh() VERSION"""

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
#output
n = x1w1x2w2 + b; n.label = 'n'
#apply tanh to the output 
o = n.tanh(); o.label = 'o'
#launch backprop on the built graph
o.backward()
```


### Tracing / visualization

For added convenience, the notebook `trace_graph.py` produces graphviz visualizations. Here we draw the neural network graph built in the example above. 

```python
from tonygrad.trace_graph import draw_dot, trace

draw_dot(o)
```

![2d neural net](output.svg)

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (*MLP*) binary classifier. This is achieved by initializing a neural net from `tonygrad.nn` module, implementing a simple svm "max-margin" binary classification loss and using *stochastic gradient descent* for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neural net](decision_boundary.png)

### License

MIT