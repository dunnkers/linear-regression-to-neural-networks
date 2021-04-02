import numpy as np
from activations import Activation, RELU, TANH
from losses import Loss, SQUARE
from node import Node, Weight
from random import choice
from itertools import islice

class Network():
    def __init__(self,
                    architecture: list[int],
                    activation: Activation = RELU(),
                    outputActivation: Activation = TANH(),
                    loss: Loss = SQUARE()):
        self.layers: list[list[Node]] = []
        for layer in architecture[:-1]:
            nodes = [Node(activation=activation) for i in range(layer)]
            self.layers.append(nodes)
        for layer in architecture[-1:]:
            nodes = [Node(activation=outputActivation) for i in range(layer)]
            self.layers.append(nodes)
        self.links: list[Weight] = []
        self.link_layers()
        self.loss = loss

    def link_layers(self):
        curr_layers = self.layers[:-1]
        next_layers = np.roll(self.layers, shift=-1)[:-1]
        for curr_layer, next_layer in zip(curr_layers, next_layers):
            self.fully_connected(curr_layer, next_layer)
    
    def fully_connected(self, layer_a: list[Node], layer_b: list[Node]):
        for a in layer_a:
            for b in layer_b:
                link = Weight(a, b)
                a.outputs.append(link)
                b.inputs.append(link)
                self.links.append(link)

    def get_loss(self, targets: list[float]):
        output_layer = self.layers[-1]
        assert(len(targets) == len(output_layer))
        loss = 0
        for node, y in zip(output_layer, targets):
            loss += self.loss.loss(node.output, y)
        return loss

    def forward(self, inputs: list[float]):
        input_layer = self.layers[0]
        assert(len(inputs) == len(input_layer))
        for node, x in zip(input_layer, inputs): # input layer
            node.output = x
        for layer in self.layers[1:]:
            for node in layer:
                node.update_output()
        return self.layers[-1] # output layer

    def backward(self, target: list[float]):
        for node, yi in zip(self.layers[-1], target):
            node.outputDer = self.loss.grad(node.output, yi)

        rng = list(reversed(range(len(self.layers))))[:-1]
        for i in rng:
            layer = self.layers[i]
            # (1) compute derivative w.r.t. total input
            for node in layer:
                node.inputDer = node.outputDer * \
                    node.activation.grad(node.totalInput)
                node.accInputDer += node.inputDer
                node.numAccumulatedDers += 1
            
            # (2) compute derivative w.r.t. weight coming into node
            for node in layer:
                for link in node.inputs:
                    link.errorDer = node.inputDer * link.src.output
                    link.accErrorDer += link.errorDer
                    link.numAccumulatedDers += 1

            if i == 1:
                continue

            prevLayer = self.layers[i - 1]
            for node in prevLayer:
                node.outputDer = 0
                for link in node.outputs:
                    node.outputDer += link.weight * \
                        link.dest.inputDer
    
    def learn(self, lr: float = 0.01):
        for layer in self.layers[1:]:
            for node in layer:
                # update bias
                if node.numAccumulatedDers > 0:
                    node.bias -= (lr / node.numAccumulatedDers) * \
                        node.accInputDer
                    node.accInputDer = 0
                    node.numAccumulatedDers = 0

                # update weights coming into this node
                for link in node.inputs:
                    if link.numAccumulatedDers > 0:
                        link.weight -= (lr / link.numAccumulatedDers) * \
                            link.accErrorDer
                        link.accErrorDer = 0
                        link.numAccumulatedDers = 0

    def sample_dataset(self, data):
        while True: yield choice(data)
    
    def sample_batch(self, data, batch_size):
        return islice(self.sample_dataset(data), batch_size)

    def fit_batch(self, batch, lr):
        loss = 0
        for x, y in batch:
            self.forward(x)
            loss += self.get_loss(y)
            self.backward(y)
        self.learn(lr=lr)
        return loss

    def fit(self, X, Y, batch_size=32, lr=0.03, max_epochs=1000, tol=1e-4):
        losses = []
        data = list(zip(X, Y))
        for epoch in range(max_epochs):
            batch = self.sample_batch(data, batch_size)
            loss = self.fit_batch(batch, lr)
            losses.append(loss)
        return losses