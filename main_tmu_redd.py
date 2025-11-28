import argparse
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

from tsetlin.utils.booleanize import booleanize_features

N_BIT = 8  # Number of bits for booleanization
N_EPOCHS = 10  # Number of training epochs

TRAIN_BUILDING = [1, 2, 3]
TEST_BUIULDING = [5]

def balance_dataset(X_train, y_train, num_per_class=1000):
    indices = []

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        chosen = np.random.choice(cls_idx, num_per_class, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices

train_files = [f"building_{building}_main_transients_train.csv" for building in TRAIN_BUILDING]
test_files = [f"building_{building}_main_transients_train.csv" for building in TEST_BUIULDING]

# Read and concatenate
train_df = pd.concat((pd.read_csv(f"model/{f}") for f in train_files), ignore_index=True)
test_df = pd.concat((pd.read_csv(f"model/{f}") for f in test_files), ignore_index=True)

X_train = train_df[["transition", "duration"]]
X_test = test_df[["transition", "duration"]]

y_train = train_df["fridge_label"]
y_test = test_df["fridge_label"]

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Normalization
X_mean = pd.concat([X_train, X_test]).mean().to_list()
X_std = pd.concat([X_train, X_test]).std().to_list()

# Balance the training dataset
max_num = min(np.bincount(y_train))

indices = balance_dataset(X_train, y_train, num_per_class=max_num)
X_train = X_train.to_numpy()[indices]
y_train = y_train[indices]

# Balance the testing dataset
max_num = min(np.bincount(y_test))
indices = balance_dataset(X_test, y_test, num_per_class=max_num)
X_test = X_test.to_numpy()[indices]
y_test = y_test[indices]

logger.info(f"Train images shape: {X_train.shape}, Train labels shape: {y_train.shape}")
logger.info(f"Test images shape: {X_test.shape}, Test labels shape: {y_test.shape}")

X_train = booleanize_features(X_train, X_mean, X_std, num_bits=N_BIT)
X_test = booleanize_features(X_test, X_mean, X_std, num_bits=N_BIT)

data = {}
data['x_train'] = np.array(X_train)
data['y_train'] = np.array(y_train)

data['x_test'] = np.array(X_test)
data['y_test'] = np.array(y_test)

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform="CPU",
        weighted_clauses=args.weighted_clauses,
        seed=42,
    )

    logger.info(f"Running {TMClassifier} for {args.epochs}")
    logger.debug(f"Input shape of training images: {data['x_train'][:args.train].shape}")
    logger.debug(f"Input shape of testing images: {data['x_test'][:args.test].shape}")

    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    data["x_train"][:args.train].astype(np.uint32),
                    data["y_train"][:args.train].astype(np.uint32),
                    metrics=["update_p"],
                )

            experiment_results["train_time"].append(benchmark1.elapsed())

            # print(res)
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"][:args.test]) == data["y_test"][:args.test]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            logger.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=200, type=int)
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--max_included_literals", default=None, type=int)
    parser.add_argument("--weighted_clauses", default=False, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train", default=-1, type=int)
    parser.add_argument("--test", default=-1, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    logger.info(results)
