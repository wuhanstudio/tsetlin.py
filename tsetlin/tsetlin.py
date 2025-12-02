import random
import sys

from tsetlin.clause import Clause
from tsetlin.utils import argmax, clip, to_int32

from tsetlin.utils.tqdm import m_tqdm

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause, N_state):

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

    def predict(self, X, return_votes=False, n_jobs=8):
        y_pred = [0] * len(X)
        n_half = int(self.n_clauses / 2)

        votes_list = []
        for i in m_tqdm(range(len(X)), desc="Predicting", ascii=True):
            Xi = X[i]
            votes = [0] * self.n_classes
            for c in range(self.n_classes):
                for j in range(n_half):
                    votes[c] += self.pos_clauses[c][j].evaluate(Xi)
                    votes[c] -= self.neg_clauses[c][j].evaluate(Xi)

            y_pred[i] = argmax(votes)

            if return_votes:
                votes_list.append(votes)

        if return_votes:
            return y_pred, votes_list
        else:
            return y_pred

    def step(self, X, y_target, T, s, return_feedback=False, threshold=-1, logger=None):
        # Pair-wise learning
        feedback = {'target': {'type-1': 0, 'type-2': 0}, 'non-target': {'type-1': 0, 'type-2': 0}}

        # Pair 1: Target class
        class_sum = 0
        pos_clauses = [0] * int(self.n_clauses / 2)
        neg_clauses = [0] * int(self.n_clauses / 2)
        for i in range(int(self.n_clauses / 2)):
            pos_clauses[i] = self.pos_clauses[y_target][i].evaluate(X)
            neg_clauses[i] = self.neg_clauses[y_target][i].evaluate(X)
            class_sum += pos_clauses[i]
            class_sum -= neg_clauses[i]

        # Clamp class_sum to [-T, T]
        class_sum = clip(class_sum, -T, T)
    
        # Calculate probabilities
        c1 = (T - class_sum) / (2 * T)

        # Update clauses for the target class
        for i in range(int(self.n_clauses / 2)):
            if (random.random() <= c1):
                # Positive Clause: Type I Feedback
                feedback_count = self.pos_clauses[y_target][i].update(X, 1, pos_clauses[i], s=s, threshold=threshold, logger=logger)
                if return_feedback:
                    feedback['target']['type-1'] += feedback_count
                if logger is not None:
                    logger.debug(f"Clause updated for target class {y_target}, clause {i}, Type I feedback count {feedback_count}")
            if (random.random() <= c1):
                # Negative Clause: Type II Feedback
                feedback_count = self.neg_clauses[y_target][i].update(X, 0, neg_clauses[i], s=s, threshold=threshold, logger=logger)
                if return_feedback:
                    feedback['target']['type-2'] += feedback_count
                if logger is not None:
                    logger.debug(f"Clause updated for target class {y_target}, clause {i}, Type II feedback count {feedback_count}")

        # Pair 2: Non-target classes
        other_class = random.choice([x for x in range(self.n_classes) if x != y_target])

        class_sum = 0
        pos_clauses = [0] * int(self.n_clauses / 2)
        neg_clauses = [0] * int(self.n_clauses / 2)
        for i in range(int(self.n_clauses / 2)):
            pos_clauses[i] = self.pos_clauses[other_class][i].evaluate(X)
            neg_clauses[i] = self.neg_clauses[other_class][i].evaluate(X)
            class_sum += pos_clauses[i]
            class_sum -= neg_clauses[i]

        # Clamp class_sum to [-T, T]
        class_sum = clip(class_sum, -T, T)

        # Calculate probabilities
        c2 = (T + class_sum) / (2 * T)
        for i in range(int(self.n_clauses / 2)):
            if (random.random() <= c2):
                # Positive Clause: Type II Feedback
                feedback_count = self.pos_clauses[other_class][i].update(X, 0, pos_clauses[i], s=s, threshold=threshold, logger=logger)
                if return_feedback:
                    feedback['non-target']['type-2'] += feedback_count
                if logger is not None:
                    logger.debug(f"Clause updated for non-target class {other_class}, clause {i}, Type II feedback count {feedback_count}")
            if (random.random() <= c2):
                # Negative Clause: Type I Feedback
                feedback_count = self.neg_clauses[other_class][i].update(X, 1, neg_clauses[i], s=s, threshold=threshold, logger=logger)
                if return_feedback:
                    feedback['non-target']['type-1'] += feedback_count
                if logger is not None:
                    logger.debug(f"Clause updated for non-target class {other_class}, clause {i}, Type I feedback count {feedback_count}")

        if return_feedback:
            return feedback

    def fit(self, X, y, T, s, epochs):
        for epoch in m_tqdm(range(epochs), ascii=True, desc="Training Epochs"):
            for i in range(len(X)):
                self.step(X[i], y[i], T=T, s=s)

    @staticmethod
    def load_model(path, threshold=-1):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        with open(path, "rb") as f:
            tm.ParseFromString(f.read())

        tm_model = Tsetlin(N_feature=tm.n_feature, N_class=tm.n_class, N_clause=tm.n_clause, N_state=tm.n_state)
        tm_model.n_classes = tm.n_class
        tm_model.n_features = tm.n_feature
        tm_model.n_clauses = tm.n_clause
        tm_model.n_states = tm.n_state

        tm_model.pos_clauses = []
        tm_model.neg_clauses = []
        for i in range(tm_model.n_classes):
            pos_clauses = []
            neg_clauses = []
            for j in range(tm_model.n_clauses // 2):
                p_clause = tm.clauses[i * tm_model.n_clauses + j * 2]
                n_clause = tm.clauses[i * tm_model.n_clauses + j * 2 + 1]

                # Set positive clauses
                pos_clause = Clause(tm_model.n_features, tm_model.n_states)
                pos_clause.set_state(p_clause.data, threshold=threshold)
                pos_clauses.append(pos_clause)

                # Set negative clauses
                neg_clause = Clause(tm_model.n_features, tm_model.n_states)
                neg_clause.set_state(n_clause.data, threshold=threshold)
                neg_clauses.append(neg_clause)

            tm_model.pos_clauses.append(pos_clauses)
            tm_model.neg_clauses.append(neg_clauses)

        return tm_model

    @staticmethod
    def load_umodel(path):
        # Load micropython model
        import tsetlin_upb2
    
        tm = None
        with open(path, "rb") as f:
            tm = tsetlin_upb2.Tsetlin.decode(f.read())
            tm.clause = tm.clause[0]

        if tm is None:
            raise ValueError("Failed to decode the model file.")

        tm_model = Tsetlin(N_feature=tm.n_feature, N_class=tm.n_class, N_clause=tm.n_clause, N_state=tm.n_state)
        tm_model.n_classes = tm.n_class
        tm_model.n_features = tm.n_feature
        tm_model.n_clauses = tm.n_clause
        tm_model.n_states = tm.n_state

        tm_model.pos_clauses = []
        tm_model.neg_clauses = []
        for i in range(tm_model.n_classes):
            pos_clauses = []
            neg_clauses = []
            for j in range(tm_model.n_clauses // 2):
                p_clause = tm.clause[(i * tm_model.n_clauses * tm_model.n_features * 2 + j * 2 * tm_model.n_features * 2): (i * tm_model.n_clauses * tm_model.n_features * 2 + j * 2 * tm_model.n_features * 2 + tm_model.n_features * 2)]
                n_clause = tm.clause[(i * tm_model.n_clauses * tm_model.n_features * 2 + j * 2 * tm_model.n_features * 2 + tm_model.n_features * 2): (i * tm_model.n_clauses * tm_model.n_features * 2 + j * 2 * tm_model.n_features * 2 + tm_model.n_features * 4)]

                # Set positive clauses
                pos_clause = Clause(tm_model.n_features, tm_model.n_states)
                pos_clause.set_state(p_clause)
                pos_clauses.append(pos_clause)

                # Set negative clauses
                neg_clause = Clause(tm_model.n_features, tm_model.n_states)
                neg_clause.set_state(n_clause)
                neg_clauses.append(neg_clause)

            tm_model.pos_clauses.append(pos_clauses)
            tm_model.neg_clauses.append(neg_clauses)

        return tm_model

    def save_model(self, path, type="training"):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        tm.n_class = self.n_classes
        tm.n_feature = self.n_features
        tm.n_clause = self.n_clauses
        tm.n_state = self.n_states

        if type not in ["training", "inference"]:
            raise ValueError("type must be either 'training' or 'inference'")
        
        if type == "training":
            tm.model_type = tsetlin_pb2.ModelType.TRAINING
        else:
            tm.model_type = tsetlin_pb2.ModelType.INFERENCE

        for i in range(self.n_classes):
            for j in range(self.n_clauses // 2):
                # Positive clauses
                pos_c = tsetlin_pb2.Clause()
                pos_c.n_feature = self.n_features
                pos_c.n_state = self.n_states

                pos_c.data.extend([to_int32(x) for x in self.pos_clauses[i][j].get_state()])

                tm.clauses.append(pos_c)

                # Negative clauses
                neg_c = tsetlin_pb2.Clause()
                neg_c.n_feature = self.n_features
                neg_c.n_state = self.n_states

                neg_c.data.extend([to_int32(x) for x in self.neg_clauses[i][j].get_state()])

                tm.clauses.append(neg_c)

        with open(path, "wb") as f:
            f.write(tm.SerializeToString())


    def save_umodel(self, path, type="training"):
        # Save micropython model
        import tsetlin_upb2
        tm = tsetlin_upb2.Tsetlin()

        tm.n_class = self.n_classes
        tm.n_feature = self.n_features
        tm.n_clause = self.n_clauses
        tm.n_state = self.n_states

        if type not in ["training", "inference"]:
            raise ValueError("type must be either 'training' or 'inference'")
        
        clause = []
        for i in range(self.n_classes):
            for j in range(self.n_clauses // 2):
                # Positive clauses
                for v in [to_int32(x) for x in self.pos_clauses[i][j].get_state()]:
                    clause.append(v)

                # Negative clauses
                for v in [to_int32(x) for x in self.neg_clauses[i][j].get_state()]:
                    clause.append(v)

        tm.clause = clause

        with open(path, "wb") as f:
            f.write(tm.encode())
