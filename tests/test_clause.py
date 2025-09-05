import random
import unittest
import numpy as np

from tsetlin.clause import Clause

class TestClause(unittest.TestCase):
    def test_evaluate(self):
        random.seed(0)
        np.random.seed(0)

        clause = Clause(N_feature=3, N_state=10)
        
        # Manually set automata states to control actions
        clause.automata[0].state = 6  # Include feature 0
        clause.automata[1].state = 5  # Exclude (NOT feature 0)

        clause.automata[2].state = 4  # Exclude feature 1
        clause.automata[3].state = 7  # Include (NOT feature 1)

        clause.automata[4].state = 6  # Include feature 2
        clause.automata[5].state = 5  # Exclude (NOT feature 2)

        # Test case where clause should evaluate to 1
        X = np.array([1, 0, 1])  # Feature 0 included, Feature 1 excluded, Feature 2 included
        output = clause.evaluate(X)
        self.assertEqual(output, 1)
        
        # Test case where clause should evaluate to 0
        X = np.array([1, 1, 1])  # Feature 1 included, which is excluded by the clause
        output = clause.evaluate(X)
        self.assertEqual(output, 0)

    def test_update(self):
        random.seed(0)
        np.random.seed(0)
        clause = Clause(N_feature=2, N_state=10)
        
        # Manually set automata states to control actions
        clause.automata[0].state = 6  # Include feature 0
        clause.automata[1].state = 5  # Exclude (NOT feature 0)

        clause.automata[2].state = 4  # Exclude feature 1
        clause.automata[3].state = 7  # Include (NOT feature 1)

        X = np.array([1, 0])  # Feature vector

        clause_output = clause.evaluate(X)  # Should be 1
        self.assertEqual(clause_output, 1)
        
        # Update the clause based on the input and target
        clause.update(X, 1, clause_output, s=3)

        # Check if automata states have been updated correctly
        self.assertTrue(clause.automata[0].state >= 6)
