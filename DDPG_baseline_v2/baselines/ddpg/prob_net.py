import numpy as np
import tensorflow as tf
from mpi4py import MPI


class Prob(object):

    def __init__(self, feature_dim, lr, sess):
        self.X_dim = feature_dim
        self.lr = lr
        self.sess = sess

        with tf.variable_scope('prob', reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(shape=(None, self.X_dim), dtype=tf.float32, name='X')
            self.I = tf.placeholder(shape=(None), dtype=tf.float32, name='I')
            self.R = tf.placeholder(shape=[], dtype=tf.float32, name='R')
            h = tf.layers.flatten(self.X)

            num_hidden = 64
            activation=tf.sigmoid

            h = activation(tf.layers.dense(h, num_hidden))
            h = activation(tf.layers.dense(h, num_hidden))
            self.P = tf.layers.dense(h, 1)
            
            self.output = tf.sigmoid(self.P)

        # Loss
        self.loss = self.R * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.P, labels=self.I))

        # Trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-4)
        params = tf.trainable_variables('prob')
        grads_and_var = self.trainer.compute_gradients(self.loss, params)
        grads, var = zip(*grads_and_var)
        grads_and_var = list(zip(grads, var))
        self.train_op = self.trainer.apply_gradients(grads_and_var)

    def train(self, X, I, R):
        """
            X: shape = (None, X_dim)
            I: shape = (None)
            R: scalar
        """
        td_map = {
            self.X: X,
            self.I: I,
            self.R: R
        }

        _, loss = self.sess.run([self.train_op, self.loss], td_map)
        #print('Meta training loss: ', loss)

    def get_P(self, X):
        """
            Get the probs given a batch of features
            X: shape = (None, X_dim)
        """
        td_map = {
            self.X: X,
        }
        output = self.sess.run(self.output, td_map)
        return output
