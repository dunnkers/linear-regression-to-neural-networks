import unittest
from activations import RELU, SIGMOID, TANH, LINEAR

class TestActivations(unittest.TestCase):
    def test_relu(self):
        relu = RELU()
        self.assertEqual(relu.func(-1), 0)
        self.assertEqual(relu.func(0), 0)
        self.assertEqual(relu.func(1), 1)
        self.assertEqual(relu.grad(-1), 0)
        self.assertEqual(relu.grad(0), 0)
        self.assertEqual(relu.grad(1), 1)
    
    def test_linear(self):
        lin = LINEAR()
        self.assertEqual(lin.func(4), 4)
        self.assertEqual(lin.grad(4), 1)

    def test_tanh(self):
        tanh = TANH()
        pass

if __name__ == '__main__':
    unittest.main()
