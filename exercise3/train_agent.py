from __future__ import print_function

import pickle
import numpy as np
import random
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(X)
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(x_train, y_train, x_valid, y_valid, history_length=1, show_distribution=False, do_data_augmentation=True):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py,
    #    the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them,
    #    you'll maybe find one_hot() useful and you may want to return X_train_unhot ... as well.
    x_train = rgb2gray(x_train)
    x_valid = rgb2gray(x_valid)

    y_train = actionArray_to_id(y_train)
    y_valid = actionArray_to_id(y_valid)

    if show_distribution:
        plt.hist(y_train)
        plt.show()

    if do_data_augmentation:
        print("... augment data")
        straight_mask = [True if y_train[i]==0 else False for i in range(y_train.shape[0])]
        n_straight = straight_mask.count(True)
        n_resample = max(0, n_straight - int(y_train.shape[0]/3))
        for i in range(n_resample):
            rand_straight = random.randint(0, y_train.shape[0]-1)
            rand_other = random.randint(0, y_train.shape[0]-1)
            while not straight_mask[rand_straight]:
                rand_straight += 1
                rand_straight %= y_train.shape[0]
            while straight_mask[rand_other]:
                rand_other += 1
                rand_other %= y_train.shape[0]

            straight_mask[rand_straight] = False
            x_train[rand_straight] = x_train[rand_other]
            y_train[rand_straight] = y_train[rand_other]


    if show_distribution and do_data_augmentation:
        plt.hist(y_train)
        plt.show()

    y_train_onehot = one_hot(y_train)
    y_valid_onehot = one_hot(y_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action.
    # Then the input states have shape (96, 96,1). Later, add a history of the last N images to your state
    # so that a state has shape (96, 96, N).
    if history_length != 1:
        num_samples_train = x_train.shape[0]
        num_samples_valid = x_valid.shape[0]
        x_train_h = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2], history_length))
        x_valid_h = np.zeros((x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], history_length))
        for i in range(num_samples_train-1, -1, -1):
            if i < history_length:
                x_train_h[i] = np.repeat(x_train[i], history_length, axis=2)
            else:
                x_train_h[i] = np.squeeze(np.stack(x_train[i - history_length:i, ...], axis=2))
        for i in range(num_samples_valid - 1, -1, -1):
            if i < history_length:
                x_valid_h[i] = np.repeat(x_valid[i], history_length, axis=2)
            else:
                x_valid_h[i] = np.squeeze(np.stack(x_valid[i - history_length:i, ...], axis=2))

        x_train = x_train_h
        x_valid = x_valid_h

    return x_train, y_train, y_train_onehot, x_valid, y_valid, y_valid_onehot


def train_model(x_train, y_train, y_train_onehot, X_valid, y_valid, y_valid_onehot, num_layers, num_filters, filter_size,
                stride, padding, maxpool, lr, batch_size, epochs, model_dir="./models", tensorboard_dir="./tensorboard",
                save=True, saveAt="agent.ckpt"):
    if save:
        # create result and model folders
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        # separate each run from another within tensorboard
        run = 0
        while True:
            if os.path.exists(tensorboard_dir + "/run{}".format(run)):
                run += 1
            else:
                tensorboard_dir += "/run{}".format(run)
                model_dir += "/run{}".format(run)
                os.makedirs(model_dir, exist_ok=True)
                break

    print("... train model")

    # TODO: specify your neural network in model.py
    agent = Model(x_train.shape[3], num_layers, num_filters, filter_size, stride, padding, maxpool, lr)
    tensorboard_eval = Evaluation(tensorboard_dir)

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard.
    #    You can watch the progress of your training in your web browser
    #
    # training loop
    with agent.sess as sess:
        # used to find best configuration with random search
        learning_curve = []
        for e in range(epochs):
            print("epoch {}/{}".format(e + 1, epochs))

            n_batches_train = x_train.shape[0] // batch_size
            n_batches_valid = X_valid.shape[0] // batch_size

            # shuffle training data and labels similar, then pick in each run batch_size samples and train the agent
            #samples = np.random.permutation(len(x_train))
            #x_train = x_train[samples]
            #y_train = y_train[samples]
            #y_train_onehot = y_train_onehot[samples]
            for batch in range(n_batches_train):
                x_batch = x_train[batch * batch_size:(batch + 1) * batch_size]
                y_batch_onehot = y_train_onehot[batch * batch_size:(batch+1) * batch_size]
                sess.run(agent.optimizer, feed_dict={agent.X: x_batch, agent.y_onehot: y_batch_onehot})

            # calculate loss on batches to reduce the size of the convolutions
            loss_train = 0.
            acc_train = 0.
            for batch in range(n_batches_train):
                x_train_batch = x_train[batch * batch_size:(batch + 1) * batch_size]
                y_train_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
                y_train_batch_onehot = y_train_onehot[batch * batch_size:(batch + 1) * batch_size]
                loss_train += sess.run(agent.loss, feed_dict={agent.X: x_train_batch, agent.y_onehot: y_train_batch_onehot})
                acc_train += sess.run(agent.accuracy, feed_dict={agent.X: x_train_batch, agent.y: y_train_batch})
            loss_eval = 0.
            acc_eval = 0.
            for batch in range(n_batches_valid):
                x_valid_batch = X_valid[batch * batch_size:(batch + 1) * batch_size]
                y_valid_batch = y_valid[batch * batch_size:(batch + 1) * batch_size]
                y_valid_batch_onehot = y_valid_onehot[batch * batch_size:(batch + 1) * batch_size]
                loss_eval += sess.run(agent.loss, feed_dict={agent.X: x_valid_batch, agent.y_onehot: y_valid_batch_onehot})
                acc_eval += sess.run(agent.accuracy, feed_dict={agent.X: x_valid_batch, agent.y: y_valid_batch})

            acc_train /= n_batches_train
            acc_eval /= n_batches_valid
            learning_curve.append(1. - acc_eval)
            if save:
                tensorboard_eval.write_episode_data(episode=e, eval_dict={"loss_train": loss_train, "loss_eval": loss_eval,
                                                                      "acc_train": acc_train, "acc_eval": acc_eval})

        # TODO: save your agent
        if save:
            fname = saveAt
            agent.save(os.path.join(model_dir, fname))
            print("Model saved in file: {:s}/{:s}".format(model_dir, fname))
        tensorboard_eval.close_session()

        return learning_curve


if __name__ == "__main__":

    # hyperparameters
    lr = 1e-4
    num_layers = 2
    num_filters = (32, 32)
    filter_size = (3, 3)
    stride = (1, 1)
    padding = ('same', 'same')
    maxpool = (True, True)
    batch_size = 64
    hl = 1
    epochs = 10

    # read data
    x_train, y_train, X_valid, y_valid = read_data(datasets_dir="./data")

    # preprocess data
    x_train, y_train, y_train_onehot, X_valid, y_valid, y_valid_onehot = preprocessing(x_train, y_train, X_valid,
                                                                                       y_valid, history_length=hl,
                                                                                       show_distribution=False,
                                                                                       do_data_augmentation=True)

    # train model (you can change the parameters!)
    train_model(x_train, y_train, y_train_onehot, X_valid, y_valid, y_valid_onehot, num_layers, num_filters,
                filter_size, stride, padding, maxpool, lr, batch_size, epochs)
