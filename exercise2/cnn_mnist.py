from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def make_model(num_filters, filter_size):
    # Convolutional Layer #1
    conv1 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])

    # Convolutional Layer #2
    conv2 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])

    # Fully-Connected Layer
    fully_connected = tf.keras.layers.Dense(units=128)

    # Softmax layer
    softmax = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

    model = tf.keras.Sequential([
        conv1,
        pool1,
        conv2,
        pool2,
        tf.keras.layers.Flatten(),
        fully_connected,
        softmax
    ])

    return model


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size=3):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    model = make_model(num_filters, filter_size)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy

    model.compile(loss=cross_entropy_loss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size, num_epochs, validation_data=(x_valid, y_valid))
    learning_curve = [1 - i for i in history.history['val_acc']]

    return learning_curve, model  # TODO: Return the validation error after each epoch(i.e learning curve)and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    loss, test_acc = model.evaluate(x_test, y_test)
    test_error = 1 - test_acc
    return test_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    lr = 0.1  # best working in exercise 2.2
    filter_size_v = [1, 3, 5, 7]

    for filter_size in filter_size_v:

        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters,
                                                   batch_size, filter_size)

        test_error = test(x_test, y_test, model)

        plt.plot(range(epochs), learning_curve, label='filter size = {:d}'.format(filter_size))

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["lr"] = lr
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["filter_size"] = filter_size
        results["learning_curve"] = learning_curve
        results["test_error"] = test_error

        path = os.path.join(args.output_path, "results")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_run_%d.json" % args.run_id)

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()

    # draw results
    plt.title('Learning curves with different filter sizes')
    plt.xlabel('epochs')
    plt.ylabel('validation error')
    plt.legend()
    plt.savefig('filter_size.jpg')
    plt.show()
