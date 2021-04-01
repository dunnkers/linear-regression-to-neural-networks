import unittest
from losses import SQUARE

class TestActivations(unittest.TestCase):
    def test_square(self):
        square = SQUARE()
        self.assertEqual(square.loss(1, 1), 0)
        self.assertEqual(square.loss(0, 2), 2)
        self.assertEqual(square.grad(1, 2), -1)

if __name__ == '__main__':
    unittest.main()
