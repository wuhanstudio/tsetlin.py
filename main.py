import argparse
from loguru import logger

import pandas as pd
from tqdm import tqdm

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features
from tsetlin.utils.math import mean, std

from sklearn.model_selection import train_test_split

iris = pd.read_csv("iris.csv")

iris['label'] = iris['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

y = iris["label"].to_numpy()
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()

# Normalization
X_mean = mean(X)
X_std = std(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=100, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=400, help="Number of states")
    parser.add_argument("--n_bit", type=int, default=8, help="Number of bits in [1, 2, 4, 8]")
    
    parser.add_argument("--T", type=int, default=30, help="Threshold T")
    parser.add_argument("--s", type=float, default=6.0, help="Specificity s")

    args = parser.parse_args()

    N_EPOCHS = args.epochs
    N_CLAUSE = args.n_clause
    N_STATE  = args.n_state

    N_BIT = args.n_bit
    if N_BIT not in {1, 2, 4, 8}:
        raise ValueError("n_bit must be one of [1, 2, 4, 8]")

    logger.debug(f"Using {N_BIT} bits for booleanization")
    logger.debug(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    logger.debug(f"Threshold T: {args.T}, Specificity s: {args.s}")

    X_bool = booleanize_features(X, X_mean, X_std, num_bits=N_BIT)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

    y_pred = tsetlin.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    for epoch in range(N_EPOCHS):
        logger.info(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
        for i in tqdm(range(len(X_train))):
            tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s)

        y_pred = tsetlin.predict(X_train)
        accuracy = sum(y_pred == y_train) / len(y_train)

    # tsetlin.fit(X_train, y_train, T=15, s=3, epochs=EPOCHS)

    logger.info("")

    # Final evaluation
    y_pred = tsetlin.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    tsetlin.save_model("tsetlin_model.pb", type="training")
    logger.info("Model saved to tsetlin_model.pb")

    logger.info("")

    # Load the model
    n_tsetlin = Tsetlin.load_model("tsetlin_model.pb")
    logger.info("Model loaded from tsetlin_model.pb")

    # Evaluate the loaded model
    n_y_pred = n_tsetlin.predict(X_test)
    accuracy = sum(n_y_pred == y_test) / len(y_test)

    logger.info(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")
