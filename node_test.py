import unittest
from node import Node, Weight
from activations import RELU

class TestNode(unittest.TestCase):
    def test_node(self):
        node = Node(bias=0.1, activation=RELU())
        node.update_output()
        self.assertEqual(RELU().func(0.1), 0.1)
        self.assertEqual(node.output, 0.1)

    def test_weight(self):
        a = Node()
        b = Node()
        weight = Weight(a, b)
        self.assertGreaterEqual(weight.weight, -0.5)
        self.assertLessEqual(weight.weight, 0.5)
    
if __name__ == '__main__':
    unittest.main()
