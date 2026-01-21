import random
random.seed(100)

import fastrand
fastrand.pcg32_seed(100)

import argparse
import numpy as np
from bitarray import bitarray

from tsetlin import Tsetlin
from tsetlin.utils.tqdm import m_tqdm
from tsetlin.utils.dataset import balance_dataset
from tsetlin.utils.booleanize import booleanize_features

from pathlib import Path
from loguru import logger

mnist_c_root = Path("mnist_c")
mnist_folders = sorted(mnist_c_root.iterdir())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--T", type=int, default=10, help="Threshold T")
    parser.add_argument("--s", type=float, default=5.0, help="Specificity s")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for compressed TM")

    args = parser.parse_args()

    N_EPOCHS = args.epochs
    N_BIT = 8

    for corruption_dir in sorted(mnist_c_root.iterdir()):

        if not corruption_dir.is_dir():
            continue

        logger.info(f"Processing corruption: {corruption_dir.name}")

        X_train = np.load(corruption_dir / "train_images.npy")   # (N, 28, 28), uint8
        y_train = np.load(corruption_dir / "train_labels.npy")   # (N,), uint8

        X_test = np.load(corruption_dir / "test_images.npy")     # (N, 28, 28), uint8
        y_test = np.load(corruption_dir / "test_labels.npy")     # (N,), uint8

        # indices = balance_dataset(X_train, y_train, num_per_class=600)
        # X_train = X_train[indices]
        # y_train = y_train[indices]

        # indices = balance_dataset(X_test, y_test, num_per_class=100)
        # X_test = X_test[indices]
        # y_test = y_test[indices]

        # Flatten images
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        X_mean = X_train.mean()
        X_std = X_train.std()

        # Normalization (not really needed for MNIST)
        X_train = X_train.astype(np.float32)
        X_train = booleanize_features(X_train, X_mean, X_std, num_bits=N_BIT)
        X_train = np.array(X_train)

        X_test = X_test.astype(np.float32)
        X_test = booleanize_features(X_test, X_mean, X_std, num_bits=N_BIT)
        X_test = np.array(X_test)

        # Convert to bitarray
        X_train = [bitarray(list(map(bool, x))) for x in X_train]
        X_test = [bitarray(list(map(bool, x))) for x in X_test]

        # Load the model
        tsetlin = Tsetlin.load_model("tsetlin_model_200_73038_8_bit.cpb")
        logger.info("Model loaded from  tsetlin_model_200_73038_8_bit.cpb")

        logger.info(f"Number of clauses: {tsetlin.n_clauses}, Number of states: {tsetlin.n_states}")
        logger.info(f"Threshold T: {args.T}, Specificity s: {args.s}")

        # Evaluate the loaded model
        y_pred = tsetlin.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)

        logger.info(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")

        for epoch in range(N_EPOCHS):
            logger.info(f"[Epoch {epoch+1}/{N_EPOCHS}] Test Accuracy: {accuracy * 100:.2f}%")

            for i in m_tqdm(range(len(X_train))):
                feedback = tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s, threshold=args.threshold)

            y_pred = tsetlin.predict(X_test)
            accuracy = sum(y_pred == y_test) / len(y_test)

        logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
