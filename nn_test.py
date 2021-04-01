import unittest
from sklearn.datasets import make_circles
import numpy as np
from nn import Network, Activations, Dataset

class TestNN(unittest.TestCase):

    def setUp(self):
        self.X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.Y = [[0], [1], [1], [0]]
        
    def test_create_network(self):
        nn = Network([2, 2, 1])
        loss = 0
        for x, y in zip(self.X, self.Y):
            nn.forward(x)
            loss += nn.get_loss(y)
        self.assertTrue(loss >= 0)

    def test_backprop(self):
        nn = Network([2, 2, 1],
            activation=Activations.RELU,
            outputActivation=Activations.LINEAR)
        ds = Dataset(self.X, self.Y)
        nn.fit(ds, lr=0.3, loss_threshold=1e-06,
            cb=lambda e, loss: print(f'epoch {e}, loss ={loss}'))
        print(nn.links)
        # print('predictions:', nn.predict(ds))
        for x, y in ds:
            print('target=',y,'output=',nn.forward(x))

    # def test_circles_ds(self):
    #     nn = Network([2, 4, 2, 1])
    #     X, y = make_circles(200, noise=0.2, factor=0.4)
    #     y = np.expand_dims(y, axis=1)
    #     ds = Dataset(X, y)
    #     nn.fit(ds, lr=0.03, loss_threshold=1e-05,
    #         cb=lambda e, loss: print(f'epoch {e}, loss ={loss}'))

if __name__ == '__main__':
    unittest.main()
