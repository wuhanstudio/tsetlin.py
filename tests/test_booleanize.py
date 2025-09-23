import unittest
import numpy as np
import tsetlin.utils.booleanize as booleanize

class TestBooleanize(unittest.TestCase):

    def test_booleanize_1_bits(self):

        num_bits = 1
        max_val = (1 << num_bits) - 1  # 2^num_bits - 1
        
        x = 0
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False])
        self.assertTrue(np.array_equal(x_bool, expected))

        # !Important: round to even to avoid bias
        x = 0.25
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 0.5
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 0.75
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 1.0
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True])
        self.assertTrue(np.array_equal(x_bool, expected))

    def test_booleanize_2_bits(self):
        num_bits = 2
        max_val = (1 << num_bits) - 1  # 2^num_bits - 1

        x = 0
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 1 / 6
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 2 / 6
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 3 / 6
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        # !Important: round to even to avoid bias
        x = 5 / 6
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 6 / 6
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

    def test_booleanize_4_bits(self):
        num_bits = 4
        max_val = (1 << num_bits) - 1  # 2^num_bits - 1

        x = 0 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 1 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 3 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 5 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, True, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 7 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, True, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 8 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 15 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([True, True, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

    def test_booleanize_8_bits(self):
        num_bits = 8
        max_val = (1 << num_bits) - 1  # 2^num_bits - 1

        x = 0 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, False, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 1 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, False, False, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 3 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, False, False, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 5 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, False, True, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 7 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, False, True, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 8 / max_val
        x_bool = booleanize(x, num_bits=num_bits)
        expected = np.array([False, False, False, False, True, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

if __name__ == '__main__':
    unittest.main()
