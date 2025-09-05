import numpy as np
from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_literals = 2 * N_feature

        # For each feature, initialize two automata (X and NOT X)
        self.automata = [Automaton(N_state, -1) for _ in range(2 * N_feature)]

        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            self.automata[2 * i].state = N_state // 2 + np.random.choice([0, 1])
            self.automata[2 * i + 1].state = N_state // 2 + np.random.choice([0, 1])

    def evaluate(self, X):
        output = 1
        for i in range(self.N_feature):
            if self.automata[2 * i].action() == 0 and X[i] == 1:
                output = 0
                break
            if self.automata[2 * i + 1].action() == 0 and X[i] == 0:
                output = 0
                break
        return output

    def update(self, X, target, clause_output, s=3):
        # Type I Feedback (Recognize patterns)
        if target == 1 and clause_output == 1:
            for i in range(self.N_feature):
                if X[i] == 1:
                    if np.random.rand() <= (s - 1) / s:
                        self.automata[2 * i].reward()
                    if np.random.rand() <= 1 / s:
                        self.automata[2 * i + 1].penalty()
                else:
                    if np.random.rand() <= (s - 1) / s:
                        self.automata[2 * i + 1].reward()
                    if np.random.rand() <= 1 / s:
                        self.automata[2 * i].penalty()
        # Type II Feedback (Erase patterns)
        elif target == 0 and clause_output == 1:
            for i in range(self.N_feature):
                if X[i] == 1:
                    if np.random.rand() <= 1 / s:
                        self.automata[2 * i].penalty()
                    if np.random.rand() <= (s - 1) / s:
                        self.automata[2 * i + 1].reward()
                else:
                    if np.random.rand() <= 1 / s:
                        self.automata[2 * i + 1].penalty()
                    if np.random.rand() <= (s - 1) / s:
                        self.automata[2 * i].reward()
