import random
import numpy as np

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause=100, N_state=400):
        self.num_features = N_feature
        self.num_classes = N_class
        self.num_clauses = N_clause
        self.num_states = N_state

        self.clauses = []
        for _ in range(N_class):
            self.clauses.append([Clause(N_feature, N_state=N_state) for _ in range(N_clause)])

    def fit(self, X, y, epochs=1):
        for epoch in range(epochs):
            for i in range(len(X)):
                self._update(X[i], y[i])

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            votes = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                for clause in self.clauses[c]:
                    clause_output = clause.evaluate(X[i])
                    votes[c] += clause_output
            y_pred.append(np.argmax(votes))
        return np.array(y_pred)

    def _update(self, X, target):
        for c in range(self.num_classes):
            for clause in self.clauses[c]:
                clause_output = clause.evaluate(X)
                clause.update(X, 1 if c == target else 0, clause_output)
