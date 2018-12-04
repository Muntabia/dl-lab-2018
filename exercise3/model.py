import tensorflow as tf


class Model:

    def __init__(self, hl=1, num_layers=2, num_filters=(32, 32), filter_size=(3, 3),
                 stride=(1, 1), padding=('same', 'same'), maxpool=(True, True), lr=0.01):
        # TODO: Define network

        self.history_length = hl
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.maxpool = maxpool
        self.learning_rate = lr


        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=[None, 96, 96, hl], name="X")
            self.y = tf.placeholder(tf.int64, shape=[None], name="y")
            self.y_onehot = tf.placeholder(tf.int64, shape=[None, 5], name="y_onehot")

        conv = tf.layers.conv2d(
            inputs=self.X,
            filters=self.num_filters[0],
            kernel_size=self.filter_size[0],
            strides=self.stride[0],
            padding=self.padding[0],
            activation=tf.nn.relu,
            name='ConvLayer0')

        pool = None
        if maxpool[0]:
            pool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=2,
                strides=2)

        for i in range(1, num_layers):
            conv = tf.layers.conv2d(
                inputs=pool if maxpool[i-1] else conv,
                filters=self.num_filters[i],
                kernel_size=self.filter_size[i],
                strides=self.stride[i],
                padding=self.padding[i],
                activation=tf.nn.relu,
                name='ConvLayer{}'.format(i))

            if maxpool[i]:
                pool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=2,
                    strides=2,
                    name='MaxPool{}'.format(i))

        # Fully-Connected Layer
        pool_flat = tf.layers.flatten(inputs=pool if maxpool[-1] else conv)
        dense = tf.layers.dense(inputs=pool_flat, units=128, activation=tf.nn.relu)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(inputs=dense, units=5)
            self.logits_unhot = tf.argmax(self.logits, axis=1)

        with tf.name_scope("eval"):
            self.accuracy = tf.contrib.metrics.accuracy(labels=self.y, predictions=self.logits_unhot)

        # TODO: Loss and optimizer
        with tf.name_scope("train"):
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot, logits=self.logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.optimizer = optimizer.minimize(self.loss)

        # TODO: Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
