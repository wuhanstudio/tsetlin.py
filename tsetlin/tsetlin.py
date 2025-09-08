import random
import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause=100, N_state=400):

        assert N_state % 2 == 0, "N_state must be even"
        assert N_clause % 2 == 0, "N_clause must be even"

        self.n_features = N_feature
        self.n_classes = N_class

        self.n_clauses = N_clause
        self.n_states = N_state

        self.pos_clauses = []
        self.neg_clauses = []
        for _ in range(N_class):
            self.pos_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])
            self.neg_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            votes = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                for j in range(int(self.n_clauses / 2)):
                    votes[c] += self.pos_clauses[c][j].evaluate(X[i])
                    votes[c] -= self.neg_clauses[c][j].evaluate(X[i])
            y_pred.append(np.argmax(votes))
        return np.array(y_pred)

    def step(self, X, y_target, T=15, s=3):
        # TODO: Pair-wise learning
        for c in range(self.n_classes):
            class_sum = 0
            for i in range(int(self.n_clauses / 2)):
                class_sum += self.pos_clauses[c][i].evaluate(X)
                class_sum -= self.neg_clauses[c][i].evaluate(X)

            # Clamp class_sum to [-T, T]
            class_sum = np.clip(class_sum, -T, T)

            c1 = (T - class_sum) / (2 * T)
            c2 = (T + class_sum) / (2 * T)

            # Update clauses for class c
            for i in range(int(self.n_clauses / 2)):
                if (y_target == c) and (np.random.rand() <= c1):
                    self.pos_clauses[c][i].update(X, 1, self.pos_clauses[c][i].evaluate(X), s=s)
                    self.neg_clauses[c][i].update(X, 1, self.neg_clauses[c][i].evaluate(X), s=s)
                elif (y_target != c) and (np.random.rand() <= c2):
                    self.pos_clauses[c][i].update(X, 0, self.pos_clauses[c][i].evaluate(X), s=s)
                    self.neg_clauses[c][i].update(X, 0, self.neg_clauses[c][i].evaluate(X), s=s)

    def fit(self, X, y, epochs=10):
        for epoch in tqdm(range(epochs)):
            for i in range(len(X)):
                self.step(X[i], y[i])
