import unittest
from network import Network, Layer
from node import Node, Weight
import numpy as np
from activations import LINEAR, RELU, SIGMOID
from itertools import cycle
from random import sample
from sklearn.datasets import make_classification

class TestLayer(unittest.TestCase):
    def test_layer(self):
        p = 2
        nodes = [Node() for i in range(p)]
        input_layer = Layer(nodes)
        output_node = Node()
        output_layer = Layer([output_node])
        for node in input_layer.nodes:
            weight = Weight(node, output_node)
            node.outputs.append(weight)
        
        input_layer.forward([10, 20])
        output_layer.compute()
        print(output_node.output)
        pass

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.network = Network([2, 1])

    def test_network(self):
        n = len(self.network.layers)
        self.assertGreater(n, 0)

    def test_loss(self):
        j = self.network.get_loss([4.])
        self.assertGreater(j, 0)

    def test_forward(self):
        yhat = self.network.forward([0, 1])
        self.assertIsNotNone(yhat)

    def test_backward(self):
        self.network.backward([1, 2])

    def test_learn(self):
        # w = [l.weight for l in self.network.links]
        self.network.learn()
        # w2 = [l.weight for l in self.network.links]
        # self.assertNotEqual(w, w2)
    
    def test_lin_reg(self):
        """ y = ax + b """
        network = Network([1, 1],
            activation=LINEAR(),
            outputActivation=LINEAR())
        X = np.linspace(0, 1, 100)
        e = lambda: 0.2 * np.random.normal()
        a, b = 3, 5
        f = lambda x: a * x + b + e()
        Y = [f(x) for x in X]
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
        network.fit(X, Y, lr=0.1)
        a_ = network.weights.item()
        b_ = network.biases[1].item()
        self.assertAlmostEqual(a_, a, places=0)
        self.assertAlmostEqual(b_, b, places=0)

    # TODO: analyze for what weights it converges/diverges.

    def test_xor(self):
        """ y = ax + b """
        network = Network([2, 2, 1],
            activation=RELU(),
            outputActivation=LINEAR())
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [[0], [1], [1], [0]]
        losses = network.fit(X, Y, lr=0.03)
        
        print(f'finished {len(losses)} epochs. loss = {losses[-1]}')
        Ŷ = network.predict(X)
        print(f'Ŷ = {Ŷ}')
        for ŷ, y in zip(Ŷ, Y):
            self.assertAlmostEqual(ŷ[0], y[0], places=0)
        pass

    def test_logistic(self):
        X, y = make_classification(
            n_features=2, n_informative=1,
            n_redundant=0, n_clusters_per_class=1,
            random_state=42)
        Y = np.expand_dims(y, axis=1)
        nn = Network([2, 1], activation=LINEAR(),
                     outputActivation=SIGMOID())
        losses = nn.fit(X, Y)
        pass

if __name__ == '__main__':
    unittest.main()
