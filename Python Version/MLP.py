import random
from Value import Value

class Neuron:
    def __init__(self,in_dim,nonlin=None):
        self.nonlin = nonlin
        self.W = [Value(random.uniform(-1,1)) for _ in range(in_dim)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self,X):
        logit = sum((w*x for w,x in zip(self.W,X))) + self.b
        if not self.nonlin:
            return logit
        elif self.nonlin == 'Tanh':
            return logit.Tanh()
        elif self.nonlin == 'Relu':
            return logit.Relu()
    
    def parameters(self):
        return self.W + [self.b]

class Layer:
    def __init__(self,in_dim,out_dim,nonlin = None):
        self.neurons = [Neuron(in_dim,nonlin) for _ in range(out_dim)]
    
    def __call__(self,X):
        outs = [neuron(X) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        p_t = []
        for neuron in self.neurons:
            p = neuron.parameters()
            p_t.extend(p)
        return p_t

class MLP:
    def __init__(self):
        self.layers = []
    
    def Linear(self,in_dim,out_dim):
        self.layers.append(Layer(in_dim,out_dim))
    
    def Tanh(self):
        for neuron in self.layers[-1].neurons:
            neuron.nonlin = 'Tanh'
    
    def Relu(self):
        for neuron in self.layers[-1].neurons:
            neuron.nonlin = 'Relu'
    
    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def parameters(self):
        p_t = []
        for layer in self.layers:
            p = layer.parameters()
            p_t.extend(p)
        return p_t
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def step(self, lr=0.01):
        for p in self.parameters():
            p.data -= lr * p.grad
    
    def __repr__(self):
        if not self.layers:
            return "MLP (Empty)"
            
        out = ["MLP ("]
        for i, layer in enumerate(self.layers):
            # Extract dimensions from the neurons
            in_dim = len(layer.neurons[0].W)
            out_dim = len(layer.neurons)
            
            # Extract activation from the first neuron (defaults to 'Linear' if None)
            activation = layer.neurons[0].nonlin or "Linear"
            
            out.append(f"  (Layer {i}): [in_dim={in_dim}, out_dim={out_dim}, activation={activation}]")
            
        out.append(f") -> Total Parameters: {len(self.parameters())}")
        return "\n".join(out)