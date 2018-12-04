import tensorflow as tf


class Evaluation:

    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir)

        self.tf_loss_train = tf.placeholder(tf.float32, name="train_loss")
        tf.summary.scalar("loss_train", self.tf_loss_train)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
        self.tf_loss_eval = tf.placeholder(tf.float32, name="eval_loss")
        tf.summary.scalar("loss_eval", self.tf_loss_eval)
        self.tf_acc_train = tf.placeholder(tf.float32, name="train_accuracy")
        tf.summary.scalar("acc_train", self.tf_acc_train)
        self.tf_acc_eval = tf.placeholder(tf.float32, name="train_accuracy")
        tf.summary.scalar("acc_eval", self.tf_acc_eval)

        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):
        # TODO: add more metrics to the summary
        summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_loss_train: eval_dict["loss_train"],
                                                                       self.tf_loss_eval: eval_dict["loss_eval"],
                                                                       self.tf_acc_train: eval_dict["acc_train"],
                                                                       self.tf_acc_eval: eval_dict["acc_eval"]})

        self.tf_writer.add_summary(summary, episode)
        self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()