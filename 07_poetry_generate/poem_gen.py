#!/usr/bin/python
# -*- coding:utf-8 -*-

import codecs
from collections import Counter

# 1.训练集准备
# 过滤
poetrys = []
with codecs.open('./poetry.txt', encoding='utf-8') as fr:
    for line in fr:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if u'（' in content or u'(' in content or u'《' in content or u'_' in content or u'[' in content:
                continue
            if len(content) < 5 or len(content) > 80:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

print '唐诗总数：', len(poetrys)
poetrys = sorted(poetrys, key=lambda x: len(x))

# 构建word和id的双向映射表
cnt = Counter(''.join(poetrys)).most_common()
words_sorted_by_freq = zip(*cnt)[0] + (' ', )
print '文字总数：', len(words_sorted_by_freq)
id2word = dict(enumerate(words_sorted_by_freq))
word2id = dict(zip(id2word.values(), id2word.keys()))
# 将诗歌转为id向量形式
poetrys_vec = [[word2id[word] for word in poetry] for poetry in poetrys]
word_nums = len(words_sorted_by_freq)

# 2.网络定义
import tensorflow as tf

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def MultiLayerRNN(cell_type='lstm', hstate_size=128, layer_nums=2, learning_rate=1e-2, batch_size=64):

    if cell_type == 'lstm':
        cell_func = tf.contrib.rnn.BasicLSTMCell
    elif cell_type == 'gru':
        cell_func = tf.contrib.rnn.GRUCell
    elif cell_type == 'rnn':
        cell_func = tf.contrib.rnn.BasicRNNCell

    x = tf.placeholder(tf.int32, [None, None]) # 第一个是batch_size；第二个是序列长度，即time_step(由于这个是不定长的，所以是None)
    y = tf.placeholder(tf.int32, [None, None])

    # 1.embedding layer
    with tf.variable_scope('embedding'):
        embeddings = tf.get_variable('embedding_matrix', [word_nums, hstate_size])
        # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # 2.定义rnn layer
    cell = cell_func(hstate_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell]*layer_nums, state_is_tuple=True)
    initial_cstate = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_cstate = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=initial_cstate)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [hstate_size, word_nums])
        b = tf.get_variable('b', [word_nums], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, hstate_size])
    y_reshaped = tf.reshape(y, [-1])

    # 3.定义output layer
    logits = tf.matmul(rnn_outputs, W) + b
    probs  = tf.nn.softmax(logits)
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # 手动控制梯度的传播
    tvars = tf.trainable_variables()
    # grads_and_tvars = tf.compute_gradients(total_loss, tvars)
    # grads = zip(*grads_and_tvars)[0]
    grads = tf.gradients(total_loss, tvars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 5)
    opti  = tf.train.AdamOptimizer(learning_rate)
    train_step = opti.apply_gradients(zip(clipped_grads, tvars))

    saver = tf.train.Saver(tf.global_variables())

    return dict(
        x = x,
        y = y,
        initial_cstate = initial_cstate,
        final_cstate = final_cstate,
        probs = probs,
        total_loss = total_loss,
        train_step = train_step,
        saver = saver
    )


import re
import numpy as np
from time import sleep

zh_p = re.compile(u'[\u4e00-\u9fa5]')

def gen_poetry(length=5, sents=4):

    def prob2word(weights):
        cum_weights = np.cumsum(weights)
        sum_weights = np.sum(weights)
        idx = np.searchsorted(cum_weights, np.random.rand(1)*sum_weights)[0]
        # wordid = np.random.choice(words_sorted_by_freq, 1, p=probs)[0] # sum of probs is not be 1
        return id2word[idx]

    reset_graph()
    g = MultiLayerRNN(batch_size=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, './poetry_gen_model-55')

        X = np.zeros((1, 1), dtype=np.int32)
        X[0, 0] = word2id['[']
        feed_dict = {g['x']: X}
        probs_, state_ = sess.run([g['probs'], g['final_cstate']], feed_dict=feed_dict)
        word = prob2word(probs_)
        poetry = []
	for i in range(sents):
	    sent = []
	    while len(sent) < length:
		if word not in ('[', ']', u'，', u'。', u',', u'.') and zh_p.match(word):
                    sent.append(word)
                X[0, 0] = word2id[word]
                feed_dict = {g['x']: X, g['initial_cstate']: state_}
                probs_, state_ = sess.run([g['probs'], g['final_cstate']], feed_dict=feed_dict)
                word = prob2word(probs_)
            if i % 2 == 0:
                sent.append(u'，')
            else:
                sent.append(u'。')
	    poetry.append(''.join(sent))
	poetry = ''.join(poetry)
        return poetry

def gen_poetry_with_head(heads, length=5):

    def prob2word(weights):
        cum_weights = np.cumsum(weights)
        sum_weights = np.sum(weights)
        idx = np.searchsorted(cum_weights, np.random.rand(1)*sum_weights)[0]
        return id2word[idx]

    reset_graph()
    g = MultiLayerRNN(batch_size=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, './poetry_gen_model-55')

        state_ = None
        poetry = []
        X = np.zeros((1, 1), dtype=np.int32)
        for i, word in enumerate(heads):
            sent = []
            while len(sent) < length:
                print word.encode('utf-8')
		# if word not in ('[', ']', u'，', u'。', u',', u'.', ' ') and zh_p.match(word):
		if word not in ('[' , ']'):
                    sent.append(word)
                X[0, 0] = word2id[word]
                feed_dict = {g['x']: X}
                if state_ is not None:
                    feed_dict[g['initial_cstate']] = state_
                probs_, state_ = sess.run([g['probs'], g['final_cstate']], feed_dict=feed_dict)
                word = prob2word(probs_)
            if i % 2 == 0:
                sent.append(u'，')
            else:
                sent.append(u'。\n')
	    poetry.append(''.join(sent))
    return poetry

