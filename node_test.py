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
        weight_b = Weight(a, b)
        self.assertNotEqual(weight.weight, weight_b.weight)
        weight_set = Weight(a, b, weight=3.)
        self.assertEqual(weight_set.weight, 3.)

    def test_lin_reg(self):
        a = Node()
        b = Node()
        w = Weight(a, b)
        a.inputs.append(w)
        b.inputs.append(w)
        inputs = [Node() for i in range(100)]
        pass
    
if __name__ == '__main__':
    unittest.main()
