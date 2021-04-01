import unittest
from network import Network

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

    def test_fit(self):
        self.network.fit(
            [[1, 2], [3, 4]],
            
        )

if __name__ == '__main__':
    unittest.main()
