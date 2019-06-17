# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:12:33 2019

    D. P. Kingma, D. J Rezende, S. Mohamed, M. Welling 
    "Semi-Supervised Learning with Deep Generative Models,"
    Advances in Neural Information Processing Systems 27(NIPS 2014).

@author: Donggeun Kwon (donggeun.kwon@gmail.com)

Cryptographic Algorithm Lab.
Graduate School of Information Security, Korea University

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tqdm

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(0)

# fully connected hidden layer
def fully_conn(x, n_hid, name):
    n_batch, n_dim = x.shape
    W = tf.get_variable(name + '_W', dtype=tf.float32, 
                        initializer=tf.truncated_normal([int(n_dim), n_hid], stddev=0.01))
    b = tf.get_variable(name + '_b', dtype=tf.float32, initializer=tf.zeros([n_hid]))
    
    return tf.matmul(x, W) + b

# ordinary Variational Autoencoder
class M1:
    def __init__(self, n_in, n_z):
        self.n_in = n_in
        self.n_z = n_z

    def _build_encoder(self, x):
        with tf.variable_scope('M1_encoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M1_encoder1'))
            mu = fully_conn(h1, self.n_z, 'M1_mu')
            sig = tf.nn.softplus(fully_conn(h1, self.n_z, 'M1_sig'))

        return mu, sig

    def _build_decoder(self, z):
        with tf.variable_scope('M1_decoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(z, int(self.n_in / 2), 'M1_decoder1'))
            y = fully_conn(h1, self.n_in, 'M1_output')

        return y

    # code edited
    def ELBO(self, mu, sig, y):
        loss_recon = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=self.x, 
                                                                    logits=y))
        loss_kld = tf.reduce_mean(tf.distributions.kl_divergence(self._get_dist(mu, sig),
                                                                 self._get_dist(tf.zeros([self.n_z]),
                                                                                tf.ones([self.n_z]))))
        return loss_recon + 0.5 * loss_kld

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in])
        self.lr = tf.placeholder(tf.float32, [])

        mu, sig = self._build_encoder(self.x)
        e = tf.random_normal(tf.shape(mu))
        self.z = mu + tf.multiply(e, sig)

        y = self._build_decoder(self.z)

        self.loss = self.ELBO(mu, sig, y)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _get_dist(self, mu, sig):
        dist = tf.distributions.Normal(loc=mu, scale=sig)
        return dist

# Conditional Variational Autoencoder
class M2:
    def __init__(self, n_in, n_z, n_cls):
        self.n_in = n_in
        self.n_z = n_z
        self.n_cls = n_cls

    def _build_classifier(self, x):
        with tf.variable_scope('M2_classifier', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M2_classifier1'))
            pred = fully_conn(h1, self.n_cls, 'M2_classifier2')
        return pred

    def _build_encoder(self, x):
        with tf.variable_scope('M2_encoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M2_encoder1'))
            mu = fully_conn(h1, self.n_z, 'M2_mu')
            sig = tf.nn.softplus(fully_conn(h1, self.n_z, 'M2_sig'))
        return mu, sig

    def _build_decoder(self, z):
        with tf.variable_scope('M2_decoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(z, int(self.n_in / 2), 'M2_decode1'))
            y = fully_conn(h1, self.n_in, 'M2_output')
        return y

    def _pathway(self, x_l, label):
        x_with_label = tf.concat([x_l, label], 1)
        mu, sig = self._build_encoder(x_with_label)

        e = tf.random_normal(tf.shape(mu))
        z = tf.multiply(e, sig)
        y = self._build_decoder(z)

        return mu, sig, z, y

    def total_loss(self, mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u):
        # loss = Loss_labelled + Loss_unlabelled + alpha * cross_entropy
        alpha = 0.1 * 100 # 0.1 * batch_size
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, 
																				  logits=self.pred_l))
        Loss_labelled = tf.reduce_mean(self.labelled_loss(mu_l, sig_l, z_l, y_l))
        Loss_unlabelled = tf.reduce_mean(self.unlabelled_loss(mu_u, sig_u, z_u, y_u))

        return Loss_labelled + Loss_unlabelled + alpha * cross_entropy

    def labelled_loss(self, mu_l, sig_l, z_l, y_l):
        kld_loss = - 0.5 * tf.reduce_sum(1. + sig_l - tf.square(mu_l) - tf.exp(sig_l), axis=-1)
        # num_classes = 10, size = 128
        logxy_loss = 128. * tf.keras.backend.binary_crossentropy(self.x_l, tf.nn.sigmoid(y_l)) - tf.log(1. / 10.)

        return tf.reshape(kld_loss, [batch_size, -1]) + logxy_loss

    def unlabelled_loss(self, mu_u, sig_u, z_u, y_u):
        label_loss = self.labelled_loss(mu_u, sig_u, z_u, y_u)
        entropy = - tf.reduce_sum(tf.nn.sigmoid(y_u) * tf.log_sigmoid(y_u))

        return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid(y_u) * label_loss, -1)) + entropy

    def build(self):
        self.x_l = tf.placeholder(tf.float32, [None, self.n_in])
        self.x_u = tf.placeholder(tf.float32, [None, self.n_in])
        self.label = tf.placeholder(tf.float32, [None, self.n_cls])
        self.lr = tf.placeholder(tf.float32, [])

        self.pred_l = self._build_classifier(self.x_l)
        self.pred_u = self._build_classifier(self.x_u)
        mu_l, sig_l, z_l, y_l = self._pathway(self.x_l, self.label)
        mu_u, sig_u, z_u, y_u = self._pathway(self.x_u, self.pred_u)

        # caclcuating loss(need to implement)
        self.loss = self.total_loss(mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.pred_u), 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def main():
	# hyper parameters
	# it is fine to change the value if you want to
	total_epoch = 1000
	batch_size = 100
	n_in = 784
	n_z1 = 128
	n_z2 = 32
	n_cls = 10
	lr = 1e-3
	mnist = read_data_sets('./MNIST-data/', one_hot=True)

	sess = tf.Session()
	m1 = M1(n_in, n_z1)
	m1.build()
	m2 = M2(n_z1, n_z2, n_cls)
	m2.build()
	sess.run(tf.global_variables_initializer())

	# TODO: train M1
	# need to implement
	losses = list()
	epochs = tqdm.trange(total_epoch//10, desc='M1-Loss')
	fig = plt.figure(1)
	fig.suptitle('M1 training', fontsize=15)
	plt.xlabel('Epochs', fontsize=10)
	plt.ylabel('Losses', fontsize=10)
	for e in epochs:
		total_loss = 0
		for i in range(int(mnist.train.num_examples//batch_size)):
			tr_x, _ = mnist.train.next_batch(batch_size)
			loss, _ = sess.run([m1.loss, m1.opt], feed_dict={m1.x:tr_x, 
															 m1.lr:lr})
			total_loss += loss
		epochs.set_description('M1 loss=%g' % total_loss)
		losses.append(total_loss)
		plt.plot(list(range(1, e+2)), losses, c='r')
		plt.pause(0.1)
		
	fig.savefig('M1-loss.jpg')

	# TODO: train M2
	# z1 = m1.z
	# m2.x = z1
	# need to implement
	epochs = tqdm.trange(total_epoch, desc='M2-Loss')
	losses = list()
	accurs = list()
	plt.rcParams["figure.figsize"] = (10,6)
	fig, ax = plt.subplots(2, 1)
	fig.suptitle('M2 training', fontsize=15)
	for e in epochs:
		total_loss = 0
		total_acc = 0
		for i in range(int(mnist.train.num_examples//batch_size)):
			xl, xu = mnist.train.next_batch(batch_size), \
					 mnist.validation.next_batch(batch_size)
			x_l = sess.run(m1.z, feed_dict={m1.x:xl[0]})
			x_u = sess.run(m1.z, feed_dict={m1.x:xu[0]})
			
			loss, _ = sess.run([m2.loss, m2.opt], feed_dict={m2.x_l:x_l, 
															 m2.x_u:x_u,
															 m2.label:xl[1],
															 m2.lr:lr*1e-2})
			total_loss += loss
			acc = sess.run(m2.accuracy, feed_dict={m2.x_u:x_l,
												   m2.label:xl[1]})
			total_acc += acc
		total_acc /= int(mnist.train.num_examples//batch_size)
		epochs.set_description('M2 loss=%g, val_acc=%g' %(total_loss, total_acc))
		losses.append(total_loss)
		accurs.append(total_acc)
		ax[0].plot(list(range(1, e+2)), losses, c='r')
		ax[1].plot(list(range(1, e+2)), accurs, c='b')
		ax[0].set_title('Training Losses')
		ax[1].set_title('Training Accuracy')
		fig.canvas.draw()
		plt.pause(0.1)
		
	fig.savefig('M2-loss_acc.jpg')
	# predict unlabeled data
	# for test dataset
	pred = 0
	for i in tqdm.tqdm(range(mnist.test.num_examples//batch_size)):
		x, y = mnist.test.next_batch(batch_size)
		x_u = sess.run(m1.z, feed_dict={m1.x:x})
		pred += sess.run(m2.accuracy, feed_dict={m2.x_u:x_u,
												 m2.label:y})

	acc = pred / (mnist.test.num_examples//batch_size)
	print('Test Accuracy: ', acc)

if __name__  == '__main__':
    main()