import unittest
from network import Network
import numpy as np
from activations import LINEAR, RELU
from itertools import cycle
from random import sample

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
        self.network.learn()
    
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
        a_ = network.links[0].weight
        b_ = network.layers[0][0].bias
        b_ = network.layers[1][0].bias
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
        for x, y in zip(X, Y):
            pred = network.forward(x)[0].output
            print(f'pred={pred}, gt={y}')
            self.assertAlmostEqual(pred, y[0], places=0)
        pass

if __name__ == '__main__':
    unittest.main()
