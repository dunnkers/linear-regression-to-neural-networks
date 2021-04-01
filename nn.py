import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from enum import Enum

class Dataset(np.ndarray):
    def __new__(cls, X, Y):
        data = list(zip(X, Y))
        self = np.asarray(data, dtype=object).view(cls)
        self.X = X
        self.Y = Y
        return self

    def generate(self, n):
        num = 0
        while num < n:
            yield (self[num % len(self)], num)
            num += 1

class FuncEnum(Enum):
    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv
    def __call__(self, *args, deriv=False, **kwargs):
        if deriv:
            return self.deriv(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)
            


class Loss(FuncEnum):
    SQUARE = (
        lambda x, y: .5 * np.square(x - y),
        lambda x, y: x - y
    )

class Node(np.ndarray):
    def __new__(cls, value=0, activation=Activations.TANH):
        self = np.asarray(value, dtype=object).view(cls)
        self.inlinks = []
        self.outlinks = []
        self.bias = 0.1
        self.totalInput = 0
        self.activation = activation

        self.outputDer = 0
        self.inputDer = 0
        self.accInputDer = 0
        self.numAccumulatedDers = 0
        return self

    def __repr__(self):
        return f'Node[output={self.item()}]'

    def update_output(self):
        self.totalInput = self.bias
        for link in self.inlinks: # input * weight
            self.totalInput += link.src * link.weight
        output = self.activation(self.totalInput.item())
        self.set_output(output)

    def set_output(self, value):
        self.put(0, value)

    def out_link(self, link):
        self.outlinks.append(link)

    def in_link(self, link):
        self.inlinks.append(link)

class Link(np.ndarray):
    def __new__(cls, src, dest):
        weight = np.random.random() - 0.5
        self = np.asarray(weight, dtype=object).view(cls)
        self.src = src
        self.dest = dest
        self.errorDer = 0
        self.accErrorDer = 0
        self.numAccumulatedDers = 0
        return self

    @property
    def weight(self):
        return self.item()

    @weight.setter
    def weight(self, value):
        self.put(0, value)

class Network(np.ndarray):
    def __new__(cls, architecture,
                     activation=Activations.RELU,
                     outputActivation=Activations.TANH,
                     loss=Loss.SQUARE):
        layers = []
        for layer in architecture[:-1]:
            nodes = [Node(activation=activation) for i in range(layer)]
            layers.append(np.array(nodes))
        for layer in architecture[-1:]:
            nodes = [Node(activation=outputActivation) for i in range(layer)]
            layers.append(np.array(nodes))
        self = np.asarray(layers, dtype=object).view(cls)
        self.links = []
        self.link_layers()
        self.loss_func = loss
        return self

    def link_layers(self):
        curr_layers = self[:-1]
        next_layers = np.roll(self, shift=-1)[:-1]
        for curr_layer, next_layer in zip(curr_layers, next_layers):
            self.link_layer(curr_layer, next_layer)

    def link_layer(self, layer_a, layer_b):
        for node_a in layer_a:
            for node_b in layer_b:
                link = Link(node_a, node_b)
                node_a.out_link(link)
                node_b.in_link(link)
                self.links.append(link)

    def get_loss(self, target, deriv=False):
        loss = 0
        for node, y in zip(self[-1], target):
            loss += self.loss_func(node.item(), y, deriv=deriv)
        return loss

    def forward(self, inputs):
        for node, x in zip(self[0], inputs): # input layer
            node.set_output(x)
        for layer in self[1:]:
            for node in layer:
                node.update_output()
        return self[-1] # output layer

    def backward(self, target):
        for node, yi in zip(self[-1], target):
            node.outputDer = self.loss_func(node.item(), yi, deriv=True)

        rng = list(reversed(range(len(self))))[:-1]
        for i in rng:
            layer = self[i]
            # (1) compute derivative w.r.t. total input
            for node in layer:
                node.inputDer = node.outputDer * \
                    node.activation(node.totalInput, deriv=True)
                node.accInputDer += node.inputDer
                node.numAccumulatedDers += 1
            
            # (2) compute derivative w.r.t. weight coming into node
            for node in layer:
                for link in node.inlinks:
                    link.errorDer = node.inputDer * link.src.item()
                    link.accErrorDer += link.errorDer
                    link.numAccumulatedDers += 1

            if i == 1:
                continue

            prevLayer = self[i - 1]
            for node in prevLayer:
                node.outputDer = 0
                for link in node.outlinks:
                    node.outputDer += link.weight * \
                        link.dest.inputDer
    
    def learn(self, lr=0.01):
        for layer in self[1:]:
            for node in layer:
                # update bias
                if node.numAccumulatedDers > 0:
                    node.bias -= lr * node.accInputDer / \
                        node.numAccumulatedDers
                    node.accInputDer = 0
                    node.numAccumulatedDers = 0

                # update weights coming into this node
                for link in node.inlinks:
                    if link.numAccumulatedDers <= 0:
                        continue
                    link.weight -= (lr / link.numAccumulatedDers) * \
                        link.accErrorDer
                    link.accErrorDer = 0
                    link.numAccumulatedDers = 0

    def fit(self, ds, epochs=100, loss_threshold=None, 
                batch_size=None,
                lr=0.03, cb=lambda x, y: 0):
        samples = len(ds)
        batch = samples if batch_size == None else batch_size
        iters = samples * epochs if loss_threshold == None else 1e10
        for data, num in ds.generate(iters):
            x, y = data
            loss = 0
            self.forward(x)
            loss += self.get_loss(y)
            self.backward(y)
            if num % batch == 0:
                self.learn(lr=lr)
            if num % batch == 0:
                cb(num/batch, loss)
                if loss_threshold != None and loss < loss_threshold:
                    break
                loss = 0              

    def predict(self, ds):
        predictions = []
        for x, y in ds:
            predictions.append(self.forward(x))
        return predictions
