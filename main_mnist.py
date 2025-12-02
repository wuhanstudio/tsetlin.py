import random
random.seed(0)

import argparse
from bitarray import bitarray

import mnist
from tsetlin import Tsetlin

from tsetlin.utils.log import log
from tsetlin.utils.tqdm import m_tqdm
from tsetlin.utils.dataset import balance_dataset

mnist.datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

X_train = mnist.train_images()
y_train = mnist.train_labels()

X_train[X_train <= 75] = 0
X_train[X_train > 75] = 1

X_test = mnist.test_images()
y_test = mnist.test_labels()

X_test[X_test <= 75] = 0
X_test[X_test > 75] = 1

indices = balance_dataset(X_train, y_train, num_per_class=100)
X_train = X_train[indices]
y_train = y_train[indices]

indices = balance_dataset(X_test, y_test, num_per_class=20)
X_test = X_test[indices]
y_test = y_test[indices]

log(f"Train images shape: {X_train.shape}, Train labels shape: {y_train.shape}")
log(f"Test images shape: {X_test.shape}, Test labels shape: {y_test.shape}")

def ask_compression():
    while True:
        choice = input("Compress and Exit? (y/n): ").strip().lower()

        if choice in ("y", "n"):
            break
        print("Please enter 'y' or 'n'.")

    if choice == "y":
        print("Yes selected")
        return True
    else:
        print("No selected")
        return False

def plot_histogram(tsetlin):
    cube = []
    for i in range(tsetlin.n_classes):
        matrix = []
        for j in range(tsetlin.n_clauses // 2):
            row = []
            p_clause = tsetlin.pos_clauses[i][j]
            n_clause = tsetlin.neg_clauses[i][j]
            for k in range(tsetlin.n_features):
                p_auto = p_clause.p_automata[k].state
                n_auto = n_clause.p_automata[k].state
                row.append(p_auto)
                row.append(n_auto)
            matrix.append(row)
        cube.append(matrix)

    import numpy as np
    import matplotlib.pyplot as plt
    cube = np.array(cube).flatten()

    fig, ax = plt.subplots()

    # Draw histogram
    counts, bins, patches = ax.hist(cube, bins=20, edgecolor='black')

    # Total number of samples
    total = counts.sum()

    # Add percentages above bars
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count == 0:
            continue  # skip empty bins
        x = (bin_left + bin_right) / 2  # center of bar
        y = count
        percent = (count / total) * 100
        ax.text(x, y, f"{percent:.2f}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram with Percentages")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=200, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=100, help="Number of states")
    
    parser.add_argument("--T", type=int, default=100, help="Threshold T")
    parser.add_argument("--s", type=float, default=5.0, help="Specificity s")

    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--compression", action='store_true')

    args = parser.parse_args()

    N_EPOCHS = args.epochs
    N_CLAUSE = args.n_clause
    N_STATE  = args.n_state

    log(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    log(f"Threshold T: {args.T}, Specificity s: {args.s}")

    # Flatten images
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Normalization (not really needed for MNIST)
    # X_train = booleanize_features(X_train, 0, 1.0, num_bits=N_BIT)
    # X_test = booleanize_features(X_test, 0, 1.0, num_bits=N_BIT)

    # Convert to bitarray
    X_train = [bitarray(list(map(bool, x))) for x in X_train]
    X_test = [bitarray(list(map(bool, x))) for x in X_test]

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=10, N_clause=N_CLAUSE, N_state=N_STATE)

    y_pred = tsetlin.predict(X_train)
    accuracy = sum(y_pred == y_train) / len(y_train)

    # plot_histogram(tsetlin)

    target_type_1_count_list = []
    target_type_2_count_list = []
    non_target_type_1_count_list = []
    non_target_type_2_count_list = []

    for epoch in range(N_EPOCHS):
        log(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
        target_type_1_count = 0
        target_type_2_count = 0
        non_target_type_1_count = 0
        non_target_type_2_count = 0
        for i in m_tqdm(range(len(X_train))):
            feedback = tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s, return_feedback=args.feedback)
            if args.feedback:
                target_type_1_count += feedback['target']['type-1']
                target_type_2_count += feedback['target']['type-2']
                non_target_type_1_count += feedback['non-target']['type-1']
                non_target_type_2_count += feedback['non-target']['type-2']
        if args.feedback:
            target_type_1_rel_change = 0.0
            target_type_2_rel_change = 0.0
            if len(target_type_1_count_list) > 0:
                target_type_1_rel_change = abs((target_type_1_count - target_type_1_count_list[-1]) / target_type_1_count_list[-1] * 100)
                target_type_2_rel_change = abs((target_type_2_count - target_type_2_count_list[-1]) / target_type_2_count_list[-1] * 100)
            non_target_type_1_rel_change = 0.0
            non_target_type_2_rel_change = 0.0
            if len(non_target_type_1_count_list) > 0:
                non_target_type_1_rel_change = abs((non_target_type_1_count - non_target_type_1_count_list[-1]) / non_target_type_1_count_list[-1] * 100)
                non_target_type_2_rel_change = abs((non_target_type_2_count - non_target_type_2_count_list[-1]) / non_target_type_2_count_list[-1] * 100)
            log(f"Target Type I Feedbacks: {target_type_1_count}, Tolerance {target_type_1_rel_change:.2f}%, Target Type II Feedbacks: {target_type_2_count}, Tolerance {target_type_2_rel_change:.2f}%")
            log(f"Non-Target Type I Feedbacks: {non_target_type_1_count}, Tolerance {non_target_type_1_rel_change:.2f}%, Non-Target Type II Feedbacks: {non_target_type_2_count}, Tolerance {non_target_type_2_rel_change:.2f}% ")
        y_pred = tsetlin.predict(X_train)
        accuracy = sum(y_pred == y_train) / len(y_train)

        if args.compression:
            if ask_compression():
                break

        # plot_histogram(tsetlin)

        if args.feedback:
            target_type_1_count_list.append(target_type_1_count)
            target_type_2_count_list.append(target_type_2_count)
            non_target_type_1_count_list.append(non_target_type_1_count)
            non_target_type_2_count_list.append(non_target_type_2_count)
    
    if args.feedback:
        import matplotlib.pyplot as plt
        epochs = list(range(1, len(target_type_1_count_list) + 1))
        plt.plot(epochs, target_type_1_count_list, label='Target Type I')
        plt.plot(epochs, target_type_2_count_list, label='Target Type II')
        plt.plot(epochs, non_target_type_1_count_list, label='Non-Target Type I')
        plt.plot(epochs, non_target_type_2_count_list, label='Non-Target Type II')
        plt.xlabel('Epoch')
        plt.ylabel('Feedback Count')
        plt.title('Feedback Counts per Epoch')
        plt.legend()
        plt.show()

    # tsetlin.fit(X_train, y_train, T=15, s=3, epochs=EPOCHS)

    log("")

    # Final evaluation
    y_pred = tsetlin.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    log(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    tsetlin.save_model("tsetlin_model.pb", type="training")
    log("Model saved to tsetlin_model.pb")

    log("")

    # Load the model
    n_tsetlin = Tsetlin.load_model("tsetlin_model.pb")
    log("Model loaded from tsetlin_model.pb")

    # Evaluate the loaded model
    n_y_pred = n_tsetlin.predict(X_test)
    accuracy = sum(n_y_pred == y_test) / len(y_test)

    log(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")
