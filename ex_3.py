import random
import sys

import numpy as np

FEATURES_NUMBER = 784
CLASSES_NUMBER = 10
HIDDEN_LAYER_SIZE = 128
MAX_VALUE = 255
ETA = 0.01
EPOCHS_NUMBER = 30

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def get_args_arr():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    return np.loadtxt(train_x, dtype=int), np.loadtxt(train_y, dtype=int), np.loadtxt(test_x, dtype=int)


def normalize_x(train_x, test_x):
    normalize_train = train_x / MAX_VALUE
    normalize_test = test_x / MAX_VALUE
    return normalize_train, normalize_test


def get_rand_matrices():
    rnd = np.random.RandomState(0)
    w1 = np.zeros((HIDDEN_LAYER_SIZE, FEATURES_NUMBER))
    temp_w1 = np.sqrt(6.0 / (HIDDEN_LAYER_SIZE + FEATURES_NUMBER))
    for i in range(HIDDEN_LAYER_SIZE):
        for j in range(FEATURES_NUMBER):
            w1[i, j] = np.float32(rnd.uniform(-temp_w1, temp_w1))

    w2 = np.zeros((CLASSES_NUMBER, HIDDEN_LAYER_SIZE))
    temp_w2 = np.sqrt(6.0 / (CLASSES_NUMBER + HIDDEN_LAYER_SIZE))
    for i in range(CLASSES_NUMBER):
        for j in range(HIDDEN_LAYER_SIZE):
            w2[i, j] = np.float32(rnd.uniform(-temp_w2, temp_w2))

    return w1, w2


def get_rand_biases():
    b1 = np.ones((HIDDEN_LAYER_SIZE, 1)) * np.sqrt((1. / FEATURES_NUMBER))
    b2 = np.ones((CLASSES_NUMBER, 1)) * np.sqrt((1. / HIDDEN_LAYER_SIZE))
    return b1, b2


def shuffle_arr(train_x, train_y):
    zip_list = list(zip(train_x, train_y))
    random.shuffle(zip_list)
    train_x, train_y = zip(*zip_list)
    return np.array(train_x), np.array(train_y)


def train_model(train_x, train_y):
    w1, w2 = get_rand_matrices()
    b1, b2 = get_rand_biases()
    for epoc in range(EPOCHS_NUMBER):
        print(epoc)
        train_x, train_y = shuffle_arr(train_x, train_y)
        for idx, x in enumerate(train_x):
            y_t = train_y[idx].T.reshape(len(train_y[idx].T), 1)
            x_t = x.T.reshape(len(x.T), 1)

            #  fprob
            z1 = np.dot(w1, x_t) + b1
            v1 = sigmoid(z1)
            z2 = np.dot(w2, v1) + b2
            v2 = softmax(z2)

            #  bprob
            g2 = v2 - y_t
            delta_w2 = np.dot(g2, v1.T)
            delta_b2 = g2
            prev_w2 = w2
            w2 = w2 - ETA * delta_w2
            b2 = b2 - ETA * delta_b2
            e1 = np.dot(prev_w2.T, g2)
            g1 = (1 - v1) * v1 * e1
            delta_w1 = np.dot(g1, x_t.T)
            delta_b1 = g1
            w1 = w1 - ETA * delta_w1
            b1 = b1 - ETA * delta_b1

    return w1, w2, b1, b2


def get_probabilities(x, w1, w2, b1, b2):
    z1 = np.dot(w1, x) + b1
    v1 = sigmoid(z1)
    z2 = np.dot(w2, v1) + b2
    v2 = softmax(z2)
    return v2.T


def test_model(test_x, w1, w2, b1, b2):
    predictions_arr = []
    for x in test_x:
        x = x.reshape(len(x), 1)
        v2 = get_probabilities(x, w1, w2, b1, b2)
        y_hat = np.argmax(v2)
        predictions_arr.append(y_hat)

    with open('test_y', 'w') as out:
        for p in predictions_arr:
            out.write(str(p) + "\n")


def one_hot_encode(train_y):
    ohe_train_y = []
    for y in train_y:
        ohe_y = np.zeros(10)
        ohe_y[int(y)] = 1
        ohe_train_y.append(ohe_y)
    return np.array(ohe_train_y)


def main():
    train_x, train_y, test_x = get_args_arr()
    train_x, test_x = normalize_x(train_x, test_x)
    train_y = one_hot_encode(train_y)
    w1, w2, b1, b2 = train_model(train_x, train_y)
    test_model(test_x, w1, w2, b1, b2)


if __name__ == "__main__":
    main()
