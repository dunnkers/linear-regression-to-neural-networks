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
        for i in range(1000):
            for x, y in zip(X, Y):
                network.forward([x])
                network.backward([y])
            network.learn(lr=0.1)
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
        data = list(zip(X, Y))
        batch_size = 3
        i = 0
        for batch in network.sample_dataset(data, batch_size):
            loss = network.fit_batch(batch, lr=0.03)
            
            a_ = [l.weight for l in network.links]
            b__ = lambda l: [n.bias for n in l]
            b_ = [b__(l) for l in network.layers]
            i += 1
            if loss < 1e-5 or i > 10000:
                break
        print(f'finished {i} epochs.')
        for x, y in zip(X, Y):
            pred = network.forward(x)[0].output
            print(f'pred={pred}, gt={y}')
            self.assertAlmostEqual(pred, y[0], places=0)

if __name__ == '__main__':
    unittest.main()
