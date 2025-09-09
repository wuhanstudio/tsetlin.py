import numpy as np
from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_literals = 2 * N_feature

        # Positive and Negative Automata for each feature
        self.p_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        self.n_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        
        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            choice = np.random.choice([0, 1])
            self.p_automata[i].state = N_state // 2 + choice
            self.n_automata[i].state = N_state // 2 + (1 - choice)

    def evaluate(self, X):
        output = 1
        for i in range(self.N_feature):
            # Include positive literal, but feature is 0
            if self.p_automata[i].action() == 1 and X[i] == 0:
                output = 0
                break
            # TODO: This may be redundant
            # Include negative literal, but feature is 1
            if self.n_automata[i].action() == 1 and X[i] == 1:
                output = 0
                break
        return output

    def update(self, X, match_target, clause_output, s):
        # TODO: Sanity Check: Both X and NOT X should not be included
        # if np.sum([ (self.p_automata[i].action()) & (self.n_automata[i].action()) for i in range(self.N_feature)]) > 0:
            # print([ (float(self.p_automata[i].state), float(self.n_automata[i].state)) for i in range(self.N_feature)])
            # raise Exception("Error: Both X and Not X are included")

        # Type I Feedback (Support patterns)
        if match_target == 1:
            # Want clause_output to be 1
            s1 = 1 / s
            s2 = (s - 1) / s

            # Erase Pattern
            # Reduce the number of included literals
            if clause_output == 0:
                for i in range(self.N_feature):
                    if np.random.rand() <= s1:
                        self.p_automata[i].penalty()
                    if np.random.rand() <= s1:
                        self.n_automata[i].penalty()

            # Recognize Pattern
            # Increase the number of included literals
            if clause_output == 1:
                for i in range(self.N_feature):
                    # Positive literal X
                    if X[i] == 1:
                        if np.random.rand() <= s2:
                            self.p_automata[i].reward()
                        if np.random.rand() <= s1:
                            self.n_automata[i].penalty()

                    # Negative literal NOT X
                    elif X[i] == 0:
                        if np.random.rand() <= s2:
                            self.n_automata[i].reward()
                        if np.random.rand() <= s1:
                            self.p_automata[i].penalty()

        # Type II Feedback (Reject Patterns)
        else:
            # Want clause_output to be 0
            if (clause_output == 1):
                for i in range(self.N_feature):
                    # TODO
                    # My implementation: Easier to overfit
                    # if (self.p_automata[i].action() == 0) and (X[i] == 0): 
                    #         self.p_automata[i].reward()
                    #         self.n_automata[i].reward()
                    # elif (self.n_automata[i].action() == 0) and (X[i] == 1):
                    #         self.p_automata[i].reward()
                    #         self.n_automata[i].reward()

                    # Original paper implementation
                    if (self.p_automata[i].action() == 0) and (X[i] == 0): 
                        self.p_automata[i].reward()
                    elif (self.n_automata[i].action() == 0) and (X[i] == 1):
                        self.n_automata[i].reward()
    
    def set_state(self, states):
        assert states.dtype == np.uint32, "States must be of type np.int32"

        assert states.ndim  == 1
        assert states.shape[0] == 2 * self.N_feature, "States must be a 1D array with shape (2 * N_features,)"

        for i in range(self.N_feature):
            self.p_automata[i].state = states[i]
            self.n_automata[i].state = states[i + self.N_feature]

    def get_state(self):
        p_states = [a.state for a in self.p_automata]
        n_states = [a.state for a in self.n_automata]

        return np.array(p_states + n_states, dtype=np.uint32)
