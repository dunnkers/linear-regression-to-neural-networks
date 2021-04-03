import unittest
from losses import SQUARE, LOGISTIC

class TestActivations(unittest.TestCase):
    def test_square(self):
        square = SQUARE()
        self.assertEqual(square.loss(1, 1), 0)
        self.assertEqual(square.loss(0, 2), 2)
        self.assertEqual(square.grad(1, 2), -1)

    def test_logistic(self):
        logistic = LOGISTIC()
        self.assertAlmostEqual(logistic.loss(
            1, 1), 0)
        self.assertGreater(logistic.loss(
            0, 1), 0)

if __name__ == '__main__':
    unittest.main()
