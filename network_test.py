import unittest
from network import Network
import numpy as np

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
        network = Network([1, 1])
        X = np.linspace(0, 1, 100)
        e = lambda: 0.2 * np.random.normal()
        a, b = 3, 5
        f = lambda x: a * x + b + e()
        Y = [f(x) for x in X]
        for i in range(10):
            for x, y in zip(X, Y):
                network.forward([x])
                network.backward([y])
            network.learn(lr=0.1)
            a_ = network.links[0].weight
            b_ = network.layers[0][0].bias
            print(f'a={a_}, b={b_}, epoch={i}')
        self.assertAlmostEqual(a_, a)
        self.assertAlmostEqual(b_, b)

    def test_xor(self):
        """ y = ax + b """
        network = Network([2, 2, 1])
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [[0], [1], [1], [0]]
        for i in range(10):
            for x, y in zip(X, Y):
                network.forward(x)
                network.backward(y)
            network.learn(lr=0.01)
            a_ = [l.weight for l in network.links]
            b__ = lambda l: [n.bias for n in l]
            b_ = [b__(l) for l in network.layers]
        for x, y in zip(X, Y):
            pred = network.forward(x)
            print(f'pred={pred[0].output}, gt={y}')
        print(f'a={a_}, b={b_}, epoch={i}')
        for pred, gt in zip(a_, [1, 1, 1, 1, 1, -2]):
            self.assertAlmostEquals(pred, gt)
        for pred, gt in zip(b_, [0, 0, 0, -2, 0]):
            self.assertAlmostEquals(pred, gt)


if __name__ == '__main__':
    unittest.main()
