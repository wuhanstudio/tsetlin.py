import argparse
from loguru import logger

import mnist
import pandas as pd
from tqdm import tqdm

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features
from tsetlin.utils.math import mean, std

from sklearn.model_selection import train_test_split

mnist.datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

X_train = mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images()
y_test = mnist.test_labels()

logger.debug(f"Train images shape: {X_train.shape}, Train labels shape: {y_train.shape}")
logger.debug(f"Test images shape: {X_test.shape}, Test labels shape: {y_test.shape}")

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

    # Flatten images
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(float) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(float) / 255.0

    # Normalization (not really needed for MNIST)
    X_train = booleanize_features(X_train[0:1000], 0, 1.0, num_bits=N_BIT)
    X_test = booleanize_features(X_test[0:100], 0, 1.0, num_bits=N_BIT)
    y_train = y_train[0:1000]
    y_test = y_test[0:100]

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=10, N_clause=N_CLAUSE, N_state=N_STATE)

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
