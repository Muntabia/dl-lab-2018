import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import argparse

import train_agent as ta

class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid = ta.read_data(datasets_dir="./data")
        self.x_train, self.y_train, self.y_train_onehot, self.x_valid, self.y_valid, self.y_valid_onehot = \
            ta.preprocessing(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        num_layers = config['num_layers']
        num_filters = ()
        filter_size = ()
        stride = ()
        padding = ()
        maxpool = ()

        for i in range(1, num_layers+1):
            num_filters += (config['num_filters{}'.format(i)],)
            filter_size += (config['filter_size{}'.format(i)],)
            stride += (config['stride{}'.format(i)],)
            padding += (config['padding{}'.format(i)],)
            maxpool += (config['maxpool{}'.format(i)],)

        lr = config['learning_rate']
        batch_size = config['batch_size']

        epochs = budget

        # TODO: train and validate your convolutional neural networks here
        learning_curve = ta.train_model(self.x_train, self.y_train, self.y_train_onehot, self.x_valid, self.y_valid,
                                        self.y_valid_onehot, num_layers, num_filters, filter_size, stride, padding,
                                        maxpool, lr, batch_size, epochs, model_dir="./models",
                                        tensorboard_dir="./tensorboard", save=False)
        print('training finished')
        validation_error = learning_curve[-1]
        # TODO: We minimize so make sure you return the validation error here
        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example
        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=1,upper=2)
        config_space.add_hyperparameter(num_layers)

        stride1 = CSH.CategoricalHyperparameter('stride1', [1, 2])
        stride2 = CSH.CategoricalHyperparameter('stride2', [1, 2])
        config_space.add_hyperparameters([stride1, stride2])

        padding1 = CSH.CategoricalHyperparameter('padding1', ['same', 'valid'])
        padding2 = CSH.CategoricalHyperparameter('padding2', ['same', 'valid'])
        config_space.add_hyperparameters([padding1, padding2])

        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=10e-4, upper=10e-1, log=True)
        config_space.add_hyperparameter(lr)

        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, log=True)
        config_space.add_hyperparameter(batch_size)

        num_filters1 = CSH.UniformIntegerHyperparameter('num_filters1', lower=8, upper=64, log=True)
        num_filters2 = CSH.UniformIntegerHyperparameter('num_filters2', lower=8, upper=64, log=True)
        config_space.add_hyperparameters([num_filters1, num_filters2])

        filter_size1 = CSH.CategoricalHyperparameter('filter_size1', [3, 5])
        filter_size2 = CSH.CategoricalHyperparameter('filter_size2', [3, 5])
        config_space.add_hyperparameters([filter_size1, filter_size2])

        maxpool1 = CSH.CategoricalHyperparameter('maxpool1', [True, False])
        maxpool2 = CSH.CategoricalHyperparameter('maxpool2', [True, False])
        config_space.add_hyperparameters([maxpool1, maxpool2])

        use_stride2 = CS.GreaterThanCondition(stride2, num_layers, 1)
        config_space.add_condition(use_stride2)

        use_padding2 = CS.GreaterThanCondition(padding2, num_layers, 1)
        config_space.add_condition(use_padding2)

        use_num_filters2 = CS.GreaterThanCondition(num_filters2, num_layers, 1)
        config_space.add_condition(use_num_filters2)

        use_filter_size2 = CS.GreaterThanCondition(filter_size2, num_layers, 1)
        config_space.add_condition(use_filter_size2)

        use_maxpool2 = CS.GreaterThanCondition(maxpool2, num_layers, 1)
        config_space.add_condition(use_maxpool2)

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=15)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])

# TODO: retrain the best configuration (called incumbent) and compute the test error
x_train, y_train, x_valid, y_valid = ta.read_data(datasets_dir="./data")
x_train, y_train, y_train_onehot, x_valid, y_valid, y_valid_onehot = ta.preprocessing(x_train, y_train, x_valid, y_valid)

num_layers = id2config[incumbent]['config']['num_layers']
num_filters = ()
filter_size = ()
stride = ()
padding = ()
maxpool = ()
for i in range(1, num_layers + 1):
    num_filters += (id2config[incumbent]['config']['num_filters{}'.format(i)],)
    filter_size += (id2config[incumbent]['config']['filter_size{}'.format(i)],)
    stride += (id2config[incumbent]['config']['stride{}'.format(i)],)
    padding += (id2config[incumbent]['config']['padding{}'.format(i)],)
    maxpool += (id2config[incumbent]['config']['maxpool{}'.format(i)],)
lr = id2config[incumbent]['config']['learning_rate']
batch_size = id2config[incumbent]['config']['batch_size']
epochs = 15

ta.train_model(x_train, y_train, y_train_onehot, x_valid, y_valid, y_valid_onehot, num_layers, num_filters, filter_size,
               stride, padding, maxpool, lr, batch_size, epochs, model_dir="./models", tensorboard_dir="./tensorboard",
               saveAt="bestAgent.ckpt")


