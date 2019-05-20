#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import numpy.random
from tensorflow.python import pywrap_tensorflow
from functools import cmp_to_key
from sklearn.metrics import roc_auc_score



class RippleNet(object):
	def __init__(self, args, n_entity, n_relation):
		self.n_entity = n_entity
		self.n_relation = n_relation
		# self.get_ent_embs()
		self._parse_args(args)
		self._build_inputs()
		self._build_embeddings()
		self._build_model()
		self._build_train()

	def _parse_args(self, args):
		self.lr = args.lr
		self.batch_size = args.batch_size
		self.dim = args.dim
		self.n_hop = args.n_hop
		self.n_memory = args.n_memory
		self.kge_weight = args.kge_weight
		self.l2_weight = args.l2_weight

	def _build_inputs(self):
		self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
		self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")

		self.memories_h = []
		self.memories_r = []
		self.memories_t = []
		for hop in range(self.n_hop):
			self.memories_h.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
			self.memories_r.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
			self.memories_t.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))
	
	def _build_embeddings(self):
		self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float32,
												shape=[self.n_entity, self.dim],
												initializer=tf.contrib.layers.xavier_initializer())
		self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float32,
												shape=[self.n_relation, self.dim],
												initializer=tf.contrib.layers.xavier_initializer())
	
	def _build_model(self):
		# self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

		# [batch size, dim]
		self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

		self.h_emb_list = []
		self.r_emb_list = []
		self.t_emb_list = []
		for i in range(self.n_hop):
			# [batch size, n_memory, dim]
			self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))
			# [batch size, n_memory, dim, dim]
			self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))
			# [batch size, n_memory, dim]
			self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

		# [batch_size, set_size, dim]
		self.set_items_embs  = tf.concat([self.h_emb_list[0], self.t_emb_list[0]], 1)
		for i in range(1,self.n_hop):
			self.set_items_embs = tf.concat([self.set_items_embs, self.t_emb_list[i]], 1)

		cell = tf.contrib.rnn.GRUCell(self.dim)
		self.cell = tf.contrib.rnn.DropoutWrapper(cell)
		self.output, self.state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.set_items_embs, dtype=tf.float32)

		# [batch_size, dim]
		self.ht = tf.squeeze(self.output[:, -1, :])
		# [batch_size, set_size]
		self.alpha = tf.sigmoid(tf.squeeze(tf.matmul(self.output,tf.expand_dims(self.ht, axis=2))))

		# [batch_size, dim]
		self.user_embeddings = tf.squeeze(tf.matmul(tf.expand_dims(self.alpha,axis=1), self.output))
		# self.user_embeddings = tf.squeeze(self.output[:, -1, :])

		self.pred = self.predict(self.user_embeddings,self.item_embeddings)
		self.pred_normalized = tf.sigmoid(self.pred)
		self.pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.pred))

		self.kge_loss = 0
		for hop in range(self.n_hop):
			# h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
			# t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
			hRt = tf.squeeze(tf.reduce_sum( (self.h_emb_list[hop] + self.r_emb_list[hop] - self.t_emb_list[hop]) ** 2, axis=2))
			self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
		self.kge_loss = -self.kge_weight * self.kge_loss

		self.l2_loss = 0
		for hop in range(self.n_hop):
			self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
			self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
			self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
		self.l2_loss = self.l2_weight * self.l2_loss

		self.loss = self.pred_loss + self.kge_loss + self.l2_loss
		

	def predict(self, user_embeddings, item_embeddings):
		# [batch_size]
		pred = tf.reduce_sum(tf.multiply(user_embeddings, item_embeddings), axis=1)
		return pred
	
	def _build_train(self):
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def train(self, sess, feed_dict):
		return sess.run([self.optimizer, self.loss, self.user_embeddings], feed_dict)

	def eval(self, sess, feed_dict):
		labels, scores = sess.run([self.labels, self.pred_normalized], feed_dict)
		auc = roc_auc_score(y_true=labels, y_score=scores)
		predictions = [1 if i >= 0.5 else 0 for i in scores]
		acc = np.mean(np.equal(predictions, labels))
		return auc, acc
