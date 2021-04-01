import numpy as np
from activations import Activation, RELU

class Node():
    def __init__(self, activation: Activation = RELU(), bias: float = 0.1):
        self.inputs: list[Weight]       = []
        self.outputs: list[Weight]      = []
        self.bias: float                = bias
        self.output: float              = 0.
        self.totalInput: float          = 0.
        self.activation: Activation     = activation
        self.outputDer: float           = 0.
        self.inputDer: float            = 0.
        self.accInputDer: float         = 0.
        self.numAccumulatedDers: float  = 0.

    def update_output(self):
        self.totalInput = self.bias
        for link in self.inputs: # input * weight
            self.totalInput += link.src.output * link.weight
        self.output = self.activation.func(self.totalInput)

class Weight():
    def __init__(self, src: Node, dest: Node, weight=None):
        self.src: Node                  = src
        self.dest: Node                 = dest
        self.weight: float              = weight or np.random.random() - 0.5
        self.errorDer: float            = 0.
        self.accErrorDer: float         = 0.
        self.numAccumulatedDers: float  = 0.