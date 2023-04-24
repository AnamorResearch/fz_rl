# # -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow.compat.v1 as tf
from more_itertools import chunked
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras.layers import Dense, Layer

tf.disable_v2_behavior()
np.set_printoptions(300)

config = ConfigProto()


class Scaling(Layer):
    def __init__(self, gamma, **kwargs):
        self.gamma = np.array([float(i) for i in gamma])
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return x * self.gamma


class Add_Const(Layer):
    def __init__(self, const, **kwargs):
        self.const = np.array([float(i) for i in const])
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return x + self.const


class GeneratorModel:
    def __init__(self, input_shape, action_space_dim, batch):
        self.input_shape = input_shape
        self.action_space_dim = action_space_dim
        self.batch = batch
        self.g = tf.Graph()
        with self.g.as_default():
            self.session = tf.Session(config=config)
            self.build_graph()
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())

    def build_graph(self):
        self.lr_gen = tf.placeholder(dtype=tf.float32)
        self.lr_value = tf.placeholder(dtype=tf.float32)

        self.s_t = tf.placeholder(tf.float32, shape=(None, self.input_shape[0]))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))
        self.a_t = tf.placeholder(tf.float32, shape=(None, self.action_space_dim))
        self.advantage = tf.placeholder(tf.float32, shape=(None, 1))  # A(s, s') = r(s) + γ * V(s') - V(s)
        self.a_p = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_old = tf.placeholder(tf.float32, shape=(None, 1))

        l_dense1 = Dense(64, activation=tf.nn.leaky_relu)(self.s_t)
        l_dense2 = Dense(64, activation=tf.nn.leaky_relu)(l_dense1)
        self.mu = Dense(self.action_space_dim, activation='sigmoid')(l_dense2)
        self.mu = Scaling([2.0, 2.0])(self.mu)
        self.mu = Add_Const([0.1, 0.1])(self.mu)
        const = 0.5 * self.action_space_dim * tf.math.log(2 * np.pi)
        sigma = tf.fill(tf.shape(self.mu), 0.05)
        log_part = 0.5 * self.action_space_dim * tf.reduce_sum(tf.math.log(sigma + 1e-5), axis=1, keepdims=True)
        log_dist = tf.reduce_sum((self.a_t - self.mu)**2 / (sigma + 1e-5), axis=1, keepdims=True)
        self.log_action_prob = -1 * (const + log_part + log_dist)

        v_dense1 = Dense(64, activation=tf.nn.leaky_relu)(self.s_t)
        v_dense2 = Dense(64, activation=tf.nn.leaky_relu)(v_dense1)
        self.v_func = Dense(1, activation='linear')(v_dense2)

        self.log_prob_old = self.a_p

        self.r_theta = tf.exp(self.log_action_prob - tf.stop_gradient(self.log_prob_old))

        self.pre_loss = tf.multiply(self.r_theta, tf.stop_gradient(self.advantage))

        self.r_clip = self.r_theta
        self.r_clip = tf.clip_by_value(self.r_clip, 0.9, 1.1)

        clipped_pre_loss = tf.multiply(self.r_clip, tf.stop_gradient(self.advantage))
        self.loss_CLIP = -1 * tf.math.minimum(clipped_pre_loss, self.pre_loss)

        self.loss_value = tf.square(self.v_func - self.r_t)
        self.entropy = 0.01 * tf.math.exp(self.log_action_prob) * self.log_action_prob

        self.loss_total = self.loss_CLIP + self.entropy

        self.pretrain_loss = -1.0 * self.log_action_prob + self.loss_value
        self.pretrain_graph = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.pretrain_loss)

        self.graph = tf.train.AdamOptimizer(learning_rate=self.lr_gen).minimize(self.loss_total)
        self.graph_value = tf.train.AdamOptimizer(learning_rate=self.lr_value).minimize(self.loss_value)

    def update(self, states, actions, rewards, log_action_probs, advantages, v_old, lr):
        idx_list = range(len(states))
        batch_idx = list(chunked(random.sample(
            idx_list, len(idx_list)), self.batch))
        loss_list = []
        for i in batch_idx:
            feed_dict_g = {self.s_t: states[i], self.r_t: rewards[i], self.advantage: advantages[i],
                           self.a_t: actions[i], self.a_p: log_action_probs[i], self.v_old: v_old[i],
                           self.lr_gen: lr, self.lr_value: lr}
            loss, loss_value, _, _ = self.session.run(
                [self.loss_total, self.loss_value, self.graph, self.graph_value], feed_dict_g)
            loss_total = loss + 0.2 * loss_value
            if np.isnan(loss_total).any():
                import sys
                print("loss nan")
                sys.exit()

            loss_list.append(loss_total.mean())

        return loss_list

    def pretrain_update(self, states, actions, rewards, log_action_probs, advantages, v_old):
        idx_list = range(len(states))
        batch_idx = list(chunked(random.sample(idx_list, len(idx_list)), 16))
        loss_list = []
        for i in batch_idx:
            feed_dict = {
                self.s_t: states[i], self.r_t: rewards[i], self.advantage: advantages[i],
                self.a_t: actions[i], self.a_p: log_action_probs[i], self.v_old: v_old[i]
            }
            loss, _ = self.session.run([self.pretrain_loss, self.pretrain_graph], feed_dict)

        loss_list.append(np.array(loss).mean())

        return loss_list

    def feed_forward(self, states, actions):
        feed_dict = {self.s_t: states, self.a_t: actions}
        log_action_prob = self.session.run(self.log_action_prob, feed_dict)
        return log_action_prob

    def log_pdf(self, loc, scale, action):
        dim = self.action_space_dim
        log_pdf = -1 * np.array([0.5 * dim * np.log(2 * np.pi) - np.sum(np.log(scale + 1e-5)) - np.sum((action - loc)**2 / (scale + 1e-5))])
        return log_pdf

    def act(self, state):
        feed_dict = {self.s_t: state}
        mu = self.session.run([self.mu], feed_dict)
        action = np.random.normal(loc=mu, scale=0.05, size=None)
        action = action[0][0]
        log_action_prob = self.log_pdf(loc=mu, scale=0.05, action=action)
        return action, log_action_prob

    def encode_states(self, states):
        encoded = self.session.run(self.encode, {self.dsae_input: states})
        return encoded

    def predict_v(self, states):
        feed_dict = {self.s_t: states}
        v_func = self.session.run(self.v_func, feed_dict)
        return v_func

    def save_model(self, path, noprint=False):
        if not noprint:
            self.saver.save(self.session, path)
            print(path)
            print('saved gen')
        else:
            self.saver.save(self.session, path)

    def load_model(self, path, noprint=False):
        if not noprint:
            print(path)
            print('startgen')
            self.saver.restore(self.session, path)
            print('restored')
        else:
            self.saver.restore(self.session, path)
