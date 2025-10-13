import sys
import random
random.seed(0)

import argparse


from iris import load_iris_X_y

from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features
from tsetlin.utils.math import mean, std

from tsetlin.utils.tqdm import m_tqdm
from tsetlin.utils.log import log

from tsetlin.utils.split import train_test_split

# Example usage
X, y = load_iris_X_y('iris.csv')

# Normalization
X_mean = mean(X)
X_std = std(X)

X_train, X_test, y_train, y_test = None, None, None, None

def objective(trial):
    n_state = trial.suggest_int("n_state", 2, 100, step=2)
    n_clause = trial.suggest_int("n_clause", 2, 100, step=2)

    T = trial.suggest_int("T", 1, n_state)
    s = trial.suggest_float("s", 1.0, 10.0, step=0.1)

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=n_clause, N_state=n_state)

    for epoch in range(N_EPOCHS):
        for i in m_tqdm(range(len(X_train))):
            tsetlin.step(X_train[i], y_train[i], T=T, s=s)

    y_pred = tsetlin.predict(X_test)
    
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    return (1.0 - accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=200, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=400, help="Number of states")
    parser.add_argument("--n_bit", type=int, default=4, help="Number of bits in [1, 2, 4, 8]")
    
    parser.add_argument("--T", type=int, default=30, help="Threshold T")
    parser.add_argument("--s", type=float, default=6.0, help="Specificity s")

    parser.add_argument("--optuna", action='store_true')
    args = parser.parse_args()

    N_EPOCHS = args.epochs

    N_BIT = args.n_bit
    if N_BIT not in {1, 2, 4, 8}:
        raise ValueError("n_bit must be one of [1, 2, 4, 8]")

    log(f"Using {N_BIT} bits for booleanization")
    X_bool = booleanize_features(X, X_mean, X_std, num_bits=N_BIT)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    if args.optuna:
        import optuna
        
        # Create a new study.
        # study = optuna.create_study()
  
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            study_name="tsetlin-machine-iris",
            load_if_exists="True"
        )
        
        # Invoke optimization of the objective function.
        study.optimize(objective, n_trials=100)  

        print(f"Best value: {study.best_value} (params: {study.best_params})")
    else:
        N_CLAUSE = args.n_clause
        N_STATE  = args.n_state

        log(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
        log(f"Threshold T: {args.T}, Specificity s: {args.s}")

        tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

        y_pred = tsetlin.predict(X_test)
        accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

        for epoch in range(N_EPOCHS):
            log(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
            for i in m_tqdm(range(len(X_train))):
                tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s)

            y_pred = tsetlin.predict(X_train)
            accuracy = sum([ 1 if pred == train else 0 for pred, train in zip(y_pred, y_train)]) / len(y_train)

        # tsetlin.fit(X_train, y_train, T=15, s=3, epochs=EPOCHS)

        log("")

        # Final evaluation
        y_pred = tsetlin.predict(X_test)
        accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

        log(f"Test Accuracy: {accuracy * 100:.2f}%")

        if sys.implementation.name != "micropython":
            # Save the model
            tsetlin.save_model("tsetlin_model.pb", type="training")
            log("Model saved to tsetlin_model.pb")

            log("")

            # Load the model
            n_tsetlin = Tsetlin.load_model("tsetlin_model.pb")
            log("Model loaded from tsetlin_model.pb")

            # Evaluate the loaded model
            n_y_pred = n_tsetlin.predict(X_test)
            accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

            log(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")
