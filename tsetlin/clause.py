
import sys
import random

rand = None
if sys.implementation.name != 'micropython':
    import fastrand
    rand = fastrand.pcg32_uniform

    from bitarray import bitarray
    from trace_pb2 import Trace
else:
    rand = random.random

from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_states = N_state
        self.N_literals = 2 * N_feature

        # Positive and Negative Automata for each feature
        self.p_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        self.n_automata = [Automaton(N_state, -1) for _ in range(N_feature)]

        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            choice = random.choice([0, 1])
            self.p_automata[i].state = N_state // 2 + choice
            self.n_automata[i].state = N_state // 2 + (1 - choice)

        self.is_micro = (sys.implementation.name == 'micropython')
        self.compress()

    def compress(self, threshold=-1):
        if not self.is_micro:
            self.p_included_mask = bitarray(self.N_feature)
            self.n_included_mask = bitarray(self.N_feature)

        # Get the index of included literals
        self.p_included_literals = []
        self.n_included_literals = []
        for i in range(self.N_feature):
            if self.p_automata[i].action == 1:
                self.p_included_literals.append(i)
                if not self.is_micro:
                    self.p_included_mask[i] = 1
            if self.n_automata[i].action == 1:
                self.n_included_literals.append(i)
                if not self.is_micro:
                    self.n_included_mask[i] = 1

        if threshold > 0:
            self.p_trainable_literals = []
            self.n_trainable_literals = []
            for i in range(self.N_feature):
                if abs(self.p_automata[i].state - self.N_states // 2 - 0.5) <= threshold:
                    self.p_trainable_literals.append(i)
                if abs(self.n_automata[i].state - self.N_states // 2 - 0.5) <= threshold:
                    self.n_trainable_literals.append(i)

    def evaluate(self, X):
        if not self.is_micro:
            # Evaluate with compression (faster)
            if not isinstance(X, bitarray):
                X = bitarray(list(map(bool, X)))
            if X & self.p_included_mask != self.p_included_mask:
                return 0
            if (~X) & self.n_included_mask != self.n_included_mask:
                return 0
        else:
            for i in self.p_included_literals:
                if X[i] == 0:
                    return 0
            for i in self.n_included_literals:
                if X[i] == 1:
                    return 0
        return 1

    def type_I_feedback(self, X, clause_output, s, threshold=-1, trace=None):
        feedback_count = 0

        # Want clause_output to be 1
        s1 = 1 / s
        s2 = (s - 1) / s

        # Erase Pattern
        # Reduce the number of included literals
        if clause_output == 0:
            # Positive literal X
            targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
            for i in targets:
                if self.p_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    transition = False
                    if self.p_automata[i].penalty():
                        transition = True
                        if not self.is_micro:
                            self.p_included_mask[i] = 0
                        elif i in self.p_included_literals:
                            self.p_included_literals.remove(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.FORGET
                        t.action = Trace.Action.PENALTY
                        t.transition = transition
                        t.state = self.p_automata[i].state
                        trace.trace.append(t)

            # Negative literal NOT X
            targets = range(self.N_feature) if threshold < 0 else self.n_trainable_literals
            for i in targets:
                if self.n_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    transition = False
                    if self.n_automata[i].penalty():
                        transition = True
                        if not self.is_micro:
                            self.n_included_mask[i] = 0
                        elif i in self.n_included_literals:
                            self.n_included_literals.remove(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.FORGET
                        t.action = Trace.Action.PENALTY
                        t.transition = transition
                        t.state = self.n_automata[i].state
                        trace.trace.append(t)


        # Recognize Pattern
        # Increase the number of included literals
        if clause_output == 1:
            # Positive literal X
            targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
            for i in targets:
                if X[i] == 1 and self.p_automata[i].state < self.N_states and rand() <= s2:
                    feedback_count += 1
                    transition = False
                    if self.p_automata[i].reward():
                        transition = True
                        if not self.is_micro:
                            self.p_included_mask[i] = 1
                        else:
                            self.p_included_literals.append(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.RECOGNIZE
                        t.action = Trace.Action.REWARD
                        t.transition = transition
                        t.state = self.p_automata[i].state
                        trace.trace.append(t)

                elif X[i] == 0 and self.p_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    transition = False
                    if self.p_automata[i].penalty():
                        transition = True
                        if not self.is_micro:
                            self.p_included_mask[i] = 0
                        elif i in self.p_included_literals:
                            self.p_included_literals.remove(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.RECOGNIZE
                        t.action = Trace.Action.PENALTY
                        t.transition = transition
                        t.state = self.p_automata[i].state
                        trace.trace.append(t)

            # Negative literal NOT X
            targets = range(self.N_feature) if threshold < 0 else self.n_trainable_literals
            for i in targets:
                if X[i] == 1 and self.n_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    transition = False
                    if self.n_automata[i].penalty():
                        transition = True
                        if not self.is_micro:
                            self.n_included_mask[i] = 0
                        elif i in self.n_included_literals:
                            self.n_included_literals.remove(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.RECOGNIZE
                        t.action = Trace.Action.PENALTY
                        t.transition = transition
                        t.state = self.n_automata[i].state
                        trace.trace.append(t)

                elif X[i] == 0 and self.n_automata[i].state < self.N_states and rand() <= s2:
                    feedback_count += 1
                    transition = False
                    if self.n_automata[i].reward():
                        transition = True
                        if not self.is_micro:
                            self.n_included_mask[i] = 1
                        elif i in self.n_included_literals:
                            self.n_included_literals.append(i)

                    if trace is not None:
                        t = Trace()
                        t.type = Trace.Type.TYPE_I
                        t.pattern = Trace.Pattern.RECOGNIZE
                        t.action = Trace.Action.REWARD
                        t.transition = transition
                        t.state = self.n_automata[i].state
                        trace.trace.append(t)

        return feedback_count

    def type_II_feedback(self, X, threshold=-1, trace=None):
        feedback_count = 0
        # Want clause_output to be 0
        targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
        for i in targets:
            if (X[i] == 0) and (self.p_included_mask[i] == 0): 
                feedback_count += 1
                transition = False
                if self.p_automata[i].reward():
                    transition = True
                    if not self.is_micro:
                        self.p_included_mask[i] = 1
                    else:
                        self.p_included_literals.append(i)

                if trace is not None:
                    t = Trace()
                    t.type = Trace.Type.TYPE_II
                    t.pattern = Trace.Pattern.NA
                    t.action = Trace.Action.REWARD
                    t.transition = transition
                    t.state = self.p_automata[i].state
                    trace.trace.append(t)

        targets = range(self.N_feature) if threshold < 0 else self.n_trainable_literals
        for i in targets:
            if (X[i] == 1) and (self.n_included_mask[i] == 0):
                feedback_count += 1
                transition = False
                if self.n_automata[i].reward():
                    transition = True
                    if not self.is_micro:
                        self.n_included_mask[i] = 1
                    else:
                        self.n_included_literals.append(i)

                if trace is not None:
                    t = Trace()
                    t.type = Trace.Type.TYPE_II
                    t.pattern = Trace.Pattern.NA
                    t.action = Trace.Action.REWARD
                    t.transition = transition
                    t.state = self.n_automata[i].state
                    trace.trace.append(t)

        return feedback_count

    def set_state(self, states, threshold=-1):
        assert len(states) == 2 * self.N_feature, "States must be a 1D array with shape (2 * N_features)"

        for i in range(self.N_feature):
            self.p_automata[i].state = states[i]
            self.p_automata[i].update()
            self.n_automata[i].state = states[i + self.N_feature]
            self.n_automata[i].update()

        self.compress(threshold=threshold)

    def get_state(self):
        p_states = [a.state for a in self.p_automata]
        n_states = [a.state for a in self.n_automata]

        return p_states + n_states
