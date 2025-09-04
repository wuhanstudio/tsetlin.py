import unittest
import numpy as np
from tsetlin.automaton import Automaton

class TestAutomaton(unittest.TestCase):
    def test_initialization(self):
        N_state = 10
        state = 5

        a = Automaton(N_state, state)

        self.assertEqual(a.N_state, N_state)
        self.assertEqual(a.state, state)

    def test_action(self):
        a = Automaton(10, 5)
        self.assertEqual(a.action(), 0)

        a = Automaton(10, 6)
        self.assertEqual(a.action(), 1)

    def test_reward(self):
        a = Automaton(10, 5)
        a.reward()
        self.assertEqual(a.state, 6)
        a.reward()
        self.assertEqual(a.state, 7)

    def test_penalty(self):
        a = Automaton(10, 6)
        a.penalty()
        self.assertEqual(a.state, 5)
        a.penalty()
        self.assertEqual(a.state, 4)
