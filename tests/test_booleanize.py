import unittest
import numpy as np
import tsetlin.utils.booleanize as booleanize

class TestBooleanize(unittest.TestCase):

    def test_booleanize(self):
        x = 0
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, False, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 1
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, False, False, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 3
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, False, False, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 5
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, False, True, False, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 7
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, False, True, True, True])
        self.assertTrue(np.array_equal(x_bool, expected))

        x = 8
        x_bool = booleanize(x)
        expected = np.array([False, False, False, False, True, False, False, False])
        self.assertTrue(np.array_equal(x_bool, expected))

if __name__ == '__main__':
    unittest.main()
