# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import numpy as np


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        with tf.name_scope('Inputs'):
            # [B, T]， 用户行为序列(User Behavior)中的 movie id 历史行为序列。T为序列长度
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None],
                                                   name='cat_his_batch_ph')  # 用户行为序列(User Behavior)中的类别id
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # user id的batch
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')  # movie id
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')  # 类别id
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')  # [B, T]
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')  # 序列真实长度
            # target_ph里面的值为：target.append([float(ss[0]), 1-float(ss[0])])
            self.target_ph = tf.placeholder(tf.float32, [None, None],
                                            name='target_ph')  # [B,2]，label 序列, 正样本对应 1, 负样本对应 0
            self.lr = tf.placeholder(tf.float64, [])  # 学习率
            self.use_negsampling = use_negsampling
            if use_negsampling:  # True
                # mid_his = numpy.zeros((n_samples, maxlen_x))
                # noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_mid_batch_ph')
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

            self.mid_sess_his_strong = tf.placeholder(tf.int32, [None, None, None],
                                                      name='mid_sess_his_strong')  # [128, 176, 27]
            self.cat_sess_his_strong = tf.placeholder(tf.int32, [None, None, None], name='cat_sess_his_strong')
            self.sess_mask_strong = tf.placeholder(tf.int32, [None, None], name='sess_mask_strong')
            self.sess_x_mask_strong = tf.placeholder(tf.int32, [None, None, None], name='sess_x_mask_strong')

            self.mid_sess_his_weak = tf.placeholder(tf.int32, [None, None, None],
                                                    name='mid_sess_his_weak')  # [128, 176, 27]
            self.cat_sess_his_weak = tf.placeholder(tf.int32, [None, None, None], name='cat_sess_his_weak')
            self.sess_mask_weak = tf.placeholder(tf.int32, [None, None], name='sess_mask_weak')
            self.sess_x_mask_weak = tf.placeholder(tf.int32, [None, None, None], name='sess_x_mask_weak')

            self.mid_sess_his_general = tf.placeholder(tf.int32, [None, None, None],
                                                       name='mid_sess_his_general')  # [128, 176, 27]
            self.cat_sess_his_general = tf.placeholder(tf.int32, [None, None, None], name='cat_sess_his_general')
            self.sess_mask_general = tf.placeholder(tf.int32, [None, None], name='sess_mask_general')
            self.sess_x_mask_general = tf.placeholder(tf.int32, [None, None, None], name='sess_x_mask_general')

            self.mid_sess_his = tf.placeholder(tf.int32, [None, None, None],
                                               name='mid_sess_his')  # [128, 176, 27]
            self.cat_sess_his = tf.placeholder(tf.int32, [None, None, None], name='cat_sess_his')
            self.sess_mask = tf.placeholder(tf.int32, [None, None], name='sess_mask')
            self.sess_x_mask = tf.placeholder(tf.int32, [None, None, None], name='sess_x_mask')

            self.mid_sess_his_general_cf = tf.placeholder(tf.int32, [None, None, None], name='mid_sess_his_general_cf')
            self.cat_sess_his_general_cf = tf.placeholder(tf.int32, [None, None, None], name='cat_sess_his_general_cf')
            self.sess_mask_general_cf = tf.placeholder(tf.int32, [None, None], name='sess_mask_general_cf')
            self.sess_x_mask_general_cf = tf.placeholder(tf.int32, [None, None, None], name='sess_x_mask_general_cf')

            self.cl_label = tf.placeholder(tf.float32, [None, None], name='cl_label_ph')
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var",
                                                      [n_uid, EMBEDDING_DIM])  # user_id的embedding weight，[543060, 18]
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)  # 直方图展示
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var,
                                                             self.uid_batch_ph)  # 从uid embedding weight 中取出 uid embedding vector

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                 self.mid_his_batch_ph)  # 输出大小为 [B, T, 18]
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)  # 取出负样本，[B, T, 5, 18]

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                           self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)  # [B, 36]
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)  # [B, T, 36]
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)  # 沿着维度1的和，实现降维，[B, 36]
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]],
                -1)  # 对于每个正样本，只用第一个负样本，[B, T, 1, 36]
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                 36])  # [B, T, 36]

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded],
                                          -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)
            self.aux_loss = self.auxiliary_loss(self.item_his_eb[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="aux")

    def build_fcn_net(self, inp, use_dice=False, cl_emb=None, cl_emb2=None):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if cl_emb:
                sim_mat = tf.matmul(cl_emb[0], tf.transpose(cl_emb[1], [1, 0]))

                cl_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.cl_label, logits=sim_mat)
                self.cl_loss = tf.reduce_mean(cl_loss)
                # cl_loss = tf.cond(tf.train.get_global_step() <= tf.constant(100000, dtype=tf.int64),
                #                   lambda: tf.constant(0.0), lambda: cl_loss)
                self.loss += 2.0 * self.cl_loss
            if cl_emb2:
                sim_mat = tf.matmul(cl_emb2[0], tf.transpose(cl_emb2[1], [1, 0]))

                cl_loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.cl_label, logits=sim_mat)
                self.cl_loss2 = tf.reduce_mean(cl_loss2)
                # cl_loss = tf.cond(tf.train.get_global_step() <= tf.constant(100000, dtype=tf.int64),
                #                   lambda: tf.constant(0.0), lambda: cl_loss)
                self.loss += 0.2 * self.cl_loss2
            # if self.use_negsampling:
            #     self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return  loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001  # 维度不变
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            strong_to_weak = inps[11]
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.mask: inps[5],
                                                       self.target_ph: inps[6],
                                                       self.seq_len_ph: inps[7],
                                                       self.lr: inps[8],
                                                       self.noclk_mid_batch_ph: inps[9],
                                                       self.noclk_cat_batch_ph: inps[10],
                                                       self.mid_sess_his_strong: strong_to_weak[0],
                                                       self.cat_sess_his_strong: strong_to_weak[1],
                                                       self.sess_mask_strong: strong_to_weak[2],
                                                       self.sess_x_mask_strong: strong_to_weak[3],
                                                       self.mid_sess_his_weak: strong_to_weak[4],
                                                       self.cat_sess_his_weak: strong_to_weak[5],
                                                       self.sess_mask_weak: strong_to_weak[6],
                                                       self.sess_x_mask_weak: strong_to_weak[7],
                                                       self.mid_sess_his_general: strong_to_weak[8],
                                                       self.cat_sess_his_general: strong_to_weak[9],
                                                       self.sess_mask_general: strong_to_weak[10],
                                                       self.sess_x_mask_general: strong_to_weak[11],
                                                       self.mid_sess_his: strong_to_weak[12],
                                                       self.cat_sess_his: strong_to_weak[13],
                                                       self.sess_mask: strong_to_weak[14],
                                                       self.sess_x_mask: strong_to_weak[15],
                                                       self.mid_sess_his_general_cf: strong_to_weak[16],
                                                       self.cat_sess_his_general_cf: strong_to_weak[17],
                                                       self.sess_mask_general_cf: strong_to_weak[18],
                                                       self.sess_x_mask_general_cf: strong_to_weak[19],
                                                       self.cl_label: np.eye(inps[0].shape[0])
                                                   })
            return loss, accuracy, aux_loss
        else:
            strong_to_weak = inps[9]
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                self.mid_sess_his_strong: strong_to_weak[0],
                self.cat_sess_his_strong: strong_to_weak[1],
                self.sess_mask_strong: strong_to_weak[2],
                self.sess_x_mask_strong: strong_to_weak[3],
                self.mid_sess_his_weak: strong_to_weak[4],
                self.cat_sess_his_weak: strong_to_weak[5],
                self.sess_mask_weak: strong_to_weak[6],
                self.sess_x_mask_weak: strong_to_weak[7],
                self.mid_sess_his_general: strong_to_weak[8],
                self.cat_sess_his_general: strong_to_weak[9],
                self.sess_mask_general: strong_to_weak[10],
                self.sess_x_mask_general: strong_to_weak[11],
                self.mid_sess_his: strong_to_weak[12],
                self.cat_sess_his: strong_to_weak[13],
                self.sess_mask: strong_to_weak[14],
                self.sess_x_mask: strong_to_weak[15],
                self.mid_sess_his_general_cf: strong_to_weak[16],
                self.cat_sess_his_general_cf: strong_to_weak[17],
                self.sess_mask_general_cf: strong_to_weak[18],
                self.sess_x_mask_general_cf: strong_to_weak[19],
                self.cl_label: np.eye(inps[0].shape[0])
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            strong_to_weak = inps[10]
            probs, loss, accuracy, aux_loss,strong_alpha,weak_alpha,kg_alpha,cf_alpha = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss,self.strong_alpha,self.weak_alpha,self.kg_alpha,self.cf_alpha],
                                                       feed_dict={
                                                           self.uid_batch_ph: inps[0],
                                                           self.mid_batch_ph: inps[1],
                                                           self.cat_batch_ph: inps[2],
                                                           self.mid_his_batch_ph: inps[3],
                                                           self.cat_his_batch_ph: inps[4],
                                                           self.mask: inps[5],
                                                           self.target_ph: inps[6],
                                                           self.seq_len_ph: inps[7],
                                                           self.noclk_mid_batch_ph: inps[8],
                                                           self.noclk_cat_batch_ph: inps[9],
                                                           self.mid_sess_his_strong: strong_to_weak[0],
                                                           self.cat_sess_his_strong: strong_to_weak[1],
                                                           self.sess_mask_strong: strong_to_weak[2],
                                                           self.sess_x_mask_strong: strong_to_weak[3],
                                                           self.mid_sess_his_weak: strong_to_weak[4],
                                                           self.cat_sess_his_weak: strong_to_weak[5],
                                                           self.sess_mask_weak: strong_to_weak[6],
                                                           self.sess_x_mask_weak: strong_to_weak[7],
                                                           self.mid_sess_his_general: strong_to_weak[8],
                                                           self.cat_sess_his_general: strong_to_weak[9],
                                                           self.sess_mask_general: strong_to_weak[10],
                                                           self.sess_x_mask_general: strong_to_weak[11],
                                                           self.mid_sess_his: strong_to_weak[12],
                                                           self.cat_sess_his: strong_to_weak[13],
                                                           self.sess_mask: strong_to_weak[14],
                                                           self.sess_x_mask: strong_to_weak[15],
                                                           self.mid_sess_his_general_cf: strong_to_weak[16],
                                                           self.cat_sess_his_general_cf: strong_to_weak[17],
                                                           self.sess_mask_general_cf: strong_to_weak[18],
                                                           self.sess_x_mask_general_cf: strong_to_weak[19],
                                                           self.cl_label: np.eye(inps[0].shape[0])
                                                       })
            return probs, loss, accuracy, aux_loss,strong_alpha,weak_alpha,kg_alpha,cf_alpha
        else:
            strong_to_weak = inps[8]
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.mid_sess_his_strong: strong_to_weak[0],
                self.cat_sess_his_strong: strong_to_weak[1],
                self.sess_mask_strong: strong_to_weak[2],
                self.sess_x_mask_strong: strong_to_weak[3],
                self.mid_sess_his_weak: strong_to_weak[4],
                self.cat_sess_his_weak: strong_to_weak[5],
                self.sess_mask_weak: strong_to_weak[6],
                self.sess_x_mask_weak: strong_to_weak[7],
                self.mid_sess_his_general: strong_to_weak[8],
                self.cat_sess_his_general: strong_to_weak[9],
                self.sess_mask_general: strong_to_weak[10],
                self.sess_x_mask_general: strong_to_weak[11],
                self.mid_sess_his: strong_to_weak[12],
                self.cat_sess_his: strong_to_weak[13],
                self.sess_mask: strong_to_weak[14],
                self.sess_x_mask: strong_to_weak[15],
                self.mid_sess_his_general_cf: strong_to_weak[16],
                self.cat_sess_his_general_cf: strong_to_weak[17],
                self.sess_mask_general_cf: strong_to_weak[18],
                self.sess_x_mask_general_cf: strong_to_weak[19],
                self.cl_label: np.eye(inps[0].shape[0])
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DIN_V2_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=att_outputs,
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                          sequence_length=self.seq_len_ph, dtype=tf.float32,
                                          scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea],
            1)
        self.build_fcn_net(inp, use_dice=True)


class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE,
                                             use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_eb, self.item_his_eb_sum], axis=-1),
                                  self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DIN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                         EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                         use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)

        self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [128, 176, 27, eb]
        self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his)
        self.mid_sess_his_eb += self.cat_sess_his_eb
        # # RNN layer(-s)，兴趣提取层（Interest Extractor Layer）
        # # with tf.name_scope('rnn_1'):
        # #     rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
        # #                                  sequence_length=self.seq_len_ph, dtype=tf.float32,
        # #                                  scope="gru1")
        # #     tf.summary.histogram('GRU_outputs', rnn_outputs)
        # #
        # # aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
        # #                                  self.noclk_item_his_eb[:, 1:, :],
        # #                                  self.mask[:, 1:], stag="gru")  # h(t)不选最后一个，e(t+1)不选第一个
        # # self.aux_loss = aux_loss_1
        #
        # # 知识图谱聚合层
        # with tf.name_scope('Attention_layer_1'):
        #     att_outputs = kg_din_fcn_attention(self.item_eb, self.mid_sess_his_eb, ATTENTION_SIZE,
        #                                        self.sess_x_mask,
        #                                        softmax_stag=1, stag='1_1', mode='SUM', return_alphas=False)
        #     final_state = din_fcn_attention(self.item_eb, att_outputs, ATTENTION_SIZE,
        #                                     self.sess_mask,
        #                                     softmax_stag=1, stag='1_2', mode='SUM', return_alphas=False)
        #     final_state = tf.squeeze(final_state, axis=1)
        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea],
            -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat,
                                                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                              use_negsampling)

        # RNN layer(-s)，兴趣提取层（Interest Extractor Layer）
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")  # h(t)不选最后一个，e(t+1)不选第一个
        self.aux_loss = aux_loss_1

        # Attention layer，兴趣进化层（Interest Evolving Layer），主要组件是 AUGRU
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        # inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


def kg_din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                         return_alphas=False, forCnn=False):
    """
        query ：候选广告，shape: [B, H], 即i_emb；
        facts ：用户历史行为，shape: [128, 176, 27, eb],[B, K, D, H]
        mask : Batch中每个行为的真实意义，shape: [128, 176, 27],[B, K, D]
    """
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters，构建一个和mask维度一样，元素都是 1 的张量，使用 tf.equal 把mask从 int 类型转成 bool 类型
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query, scope=stag)
    # 以下计算注意力方式和din相同
    queries = tf.tile(query, [1, tf.shape(facts)[1] * tf.shape(facts)[2]])
    queries = tf.reshape(queries, tf.shape(facts))  # queries 先 reshape 成和 keys 相同的大小: [B, K, D, H]，不知道填什么的时候用-1填充
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # 按倒数第一个维度拼接,[B, K, D, H * 4]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1], tf.shape(facts)[2]])  # [B, 1, K, D]
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    # 由于一个 Batch 中的用户行为序列不一定都相同, 其真实长度保存在 keys_length 中
    # 所以这里要产生 masks 来选择真正的历史行为
    # tf.sequence_mask(2, 6)  即为array[ True  True False False False False]
    # key_masks = tf.sequence_mask([1, 2, 3], 4),即为[[ True False False False] [ True  True False False] [ True  True  True False]]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, K, D]
    # 选出真实的历史行为, 而对于那些填充的结果, 适用 paddings 中的值来表示
    # padddings 中使用巨大的负值, 后面计算 softmax 时, e^{x} 结果就约等于 0
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)  # 创建一个tensor，维度与scores相同，值为(-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, K, D]，对scores进行mask，被mask的值采用(-2 ** 32 + 1)进行替换

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)

    # Weighted sum
    if mode == 'SUM':
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1], tf.shape(facts)[2], 1])  # [B,K,D,1]
        output = tf.reduce_sum(scores * facts, axis=2)  # [B,K,H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1], tf.shape(facts)[2]])  # [B, K, D]
        output = facts * tf.expand_dims(scores, -1)  # [B, K, D, H]
        output = tf.reshape(output, tf.shape(facts))  # [B, K, D, H]
    if return_alphas:
        return output, scores
    return output


def mhsa_inter_triangle_aggregation(inputs, num_units, num_heads, dropout_rate, name="",
                                    is_training=True, is_layer_norm=True, keys_missing_value=0, EMBEDDING_DIM=18):
    """
    Args:
        inputs: [B*T, TRIANGLE_NUM, H]
    Output:
        outputs: [B*T, TRIANGLE_NUM, H]
    """
    x_shape = inputs.get_shape().as_list()
    keys_empty = tf.reduce_sum(inputs, axis=2)  # [B*T, TRIANGLE_NUM]
    keys_empty_cond = tf.equal(keys_empty, keys_missing_value)
    keys_empty_cond = tf.expand_dims(keys_empty_cond, 2)
    print inputs.get_shape()
    print keys_empty.get_shape()
    print keys_empty_cond.get_shape()
    print num_heads
    print x_shape
    keys_empty_cond = tf.tile(keys_empty_cond, [num_heads, 1, tf.shape(inputs)[1]])  # [B*T, TRIANGLE_NUM, TRIANGLE_NUM]

    Q_K_V = tf.layers.dense(inputs, 3 * num_units)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]

    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [B*T*num_heads, TRIANGLE_NUM, TRIANGLE_NUM]
    align = outputs / (36 ** 0.5)
    # Diag mask
    diag_val = tf.ones_like(align[0, :, :])  # [TRIANGLE_NUM, TRIANGLE_NUM]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [TRIANGLE_NUM, TRIANGLE_NUM]
    key_masks = tf.tile(tf.expand_dims(tril, 0),
                        [tf.shape(align)[0], 1, 1])  # [B*T*num_heads, TRIANGLE_NUM, TRIANGLE_NUM]
    paddings = tf.ones_like(key_masks) * (-2 ** 32 + 1)

    outputs = tf.where(tf.equal(key_masks, 0), paddings, align)
    outputs = tf.where(keys_empty_cond, paddings, outputs)
    outputs = tf.nn.softmax(outputs)  # [B*T, TRIANGLE_NUM, TRIANGLE_NUM]
    # Attention Matmul
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [B*T, TRIANGLE_NUM, num_units]

    outputs = tf.layers.dense(outputs, num_units / 2)  # [B*T, TRIANGLE_NUM, num_units/2]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    print outputs.get_shape()
    print inputs.get_shape()
    outputs += inputs
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)

    outputs1 = tf.layers.dense(outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
    outputs1 = tf.layers.dense(outputs1, EMBEDDING_DIM)
    outputs = outputs1 + outputs
    return outputs


def layer_norm(inputs, name, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(name + 'gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable(name + 'beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def general_attention(query, key):
    """
    :param query: [batch_size, None, query_size]
    :param key:   [batch_size, time, key_size]
    :return:      [batch_size, None, time]
        query_size should keep the same dim with key_size
    """
    # [batch_size, None, time]
    align = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
    # scale (optional)
    align = align / (key.get_shape().as_list()[-1] ** 0.5)
    return align


def soft_max_weighted_sum(align, value, key_masks, drop_out, is_training, future_binding=False):
    """
    :param align:           [batch_size, None, time]
    :param value:           [batch_size, time, units]
    :param key_masks:       [batch_size, None, time]
                            2nd dim size with align
    :param drop_out:
    :param is_training:
    :param future_binding:  TODO: only support 2D situation at present
    :return:                weighted sum vector
                            [batch_size, None, units]
    """
    # exp(-large) -> 0
    paddings = tf.fill(tf.shape(align), float('-inf'))
    # [batch_size, None, time]
    align = tf.where(key_masks, align, paddings)

    if future_binding:
        length = tf.reshape(tf.shape(value)[1], [-1])
        # [time, time]
        lower_tri = tf.ones(tf.concat([length, length], axis=0))
        # [time, time]
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
        # [batch_size, time, time]
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        # [batch_size, time, time]
        align = tf.where(tf.equal(masks, 0), paddings, align)

    # soft_max and dropout
    # [batch_size, None, time]
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, drop_out, training=is_training)
    # weighted sum
    # [batch_size, None, units]
    return tf.matmul(align, value)


# def layer_norm(inputs, epsilon=1e-8):
#     mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
#     normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
#
#     params_shape = inputs.get_shape()[-1:]
#     gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
#     beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
#
#     outputs = gamma * normalized + beta
#     return outputs


def self_multi_head_attn(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    print('V_.get_shape()', V_.get_shape().as_list())
    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [hN, T, T]
    align = outputs / (36 ** 0.5)
    # align = general_attention(Q_, K_)
    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [T, T]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() # [T, T] for tensorflow140
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, T, T]
    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    # output linear
    outputs = tf.layers.dense(outputs, num_units)

    # drop_out before residual and layernorm
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs  # (N, T_q, C)
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # (N, T_q, C)

    return outputs


def self_multi_head_attn_v2(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    print('V_.get_shape()', V_.get_shape().as_list())
    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [hN, T, T]
    align = outputs / (36 ** 0.5)
    # align = general_attention(Q_, K_)
    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [T, T]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() # [T, T] for tensorflow140
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, T, T]
    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    # Restore shape
    outputs1 = tf.split(outputs, num_heads, axis=0)
    outputs2 = []
    for head_index, outputs3 in enumerate(outputs1):
        outputs3 = tf.layers.dense(outputs3, num_units)
        outputs3 = tf.layers.dropout(outputs3, dropout_rate, training=is_training)
        outputs3 += inputs
        print("outputs3.get_shape()", outputs3.get_shape())
        if is_layer_norm:
            outputs3 = layer_norm(outputs3, name=name + str(head_index))  # (N, T_q, C)
        outputs2.append(outputs3)

    # drop_out before residual and layernorm
    # outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    # outputs += inputs  # (N, T_q, C)
    # Normalize
    #  if is_layer_norm:
    #       outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs2


def din_attention_new(query, facts, context_his_eb, on_size, mask, stag='null', mode='SUM', softmax_stag=1,
                      time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    if context_his_eb is None:
        queries = queries
    else:
        queries = tf.concat([queries, context_his_eb], axis=-1)
    queries = tf.layers.dense(queries, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr1')
    # queries = prelu(queries, scope=stag+'dmr_prelu')
    # din_all = tf.concat([queries, facts], axis=-1)
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


class Model_KISDAN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_KISDAN, self).__init__(n_uid, n_mid, n_cat,
                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                          use_negsampling)

        self.mid_sess_his_eb_strong = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                             self.mid_sess_his_strong)  # [128, 176, 27, eb]
        self.cat_sess_his_eb_strong = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his_strong)
        self.mid_sess_his_eb_strong += self.cat_sess_his_eb_strong

        self.mid_sess_his_eb_weak = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                           self.mid_sess_his_weak)  # [128, 176, 27, eb]
        self.cat_sess_his_eb_weak = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his_weak)
        self.mid_sess_his_eb_weak += self.cat_sess_his_eb_weak

        self.mid_sess_his_eb_general = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                              self.mid_sess_his_general)  # [128, 176, 27, eb]
        self.cat_sess_his_eb_general = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his_general)
        self.mid_sess_his_eb_general += self.cat_sess_his_eb_general

        self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                      self.mid_sess_his)  # [128, 176, 27, eb]
        self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his)
        self.mid_sess_his_eb += self.cat_sess_his_eb

        self.mid_sess_his_eb_general_cf = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                 self.mid_sess_his_general_cf)  # [128, 176, 27, eb]
        self.cat_sess_his_eb_general_cf = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_sess_his_general_cf)
        self.mid_sess_his_eb_general_cf += self.cat_sess_his_eb_general_cf

        self.mid_his_batch_embedded_origin = self.mid_his_batch_embedded + self.cat_his_batch_embedded
        # 知识图谱聚合层
        with tf.name_scope('Attention_layer_1'):
            # att_outputs = kg_din_fcn_attention(self.item_eb, self.mid_sess_his_eb, ATTENTION_SIZE,
            #                                    self.sess_x_mask,
            #                                    softmax_stag=1, stag='1_9', mode='SUM', return_alphas=False)
            # # att_outputs = mhsa_inter_triangle_aggregation(att_outputs,
            # #                                                      num_units=EMBEDDING_DIM * 2,
            # #                                                      num_heads=4,
            # #                                                      dropout_rate=0.,
            # #                                                      is_training=True,
            # #                                                      name="att_outputs")
            # final_state = din_fcn_attention(self.item_eb, att_outputs, ATTENTION_SIZE,
            #                                 self.sess_mask,
            #                                 softmax_stag=1, stag='1_10', mode='SUM', return_alphas=False)
            # final_state = tf.squeeze(final_state, axis=1)
            # final_state_query = final_state
            att_outputs_strong = kg_din_fcn_attention(self.item_eb, self.mid_sess_his_eb_strong, ATTENTION_SIZE,
                                                      self.sess_x_mask_strong,
                                                      softmax_stag=1, stag='1_1', mode='SUM', return_alphas=False)
            # att_outputs_strong = mhsa_inter_triangle_aggregation(att_outputs_strong,
            #                                                      num_units=EMBEDDING_DIM * 2,
            #                                                      num_heads=4,
            #                                                      dropout_rate=0.,
            #                                                      is_training=True,
            #                                                      name="strong")
            final_state_strong,self.strong_alpha = din_fcn_attention(self.item_eb, att_outputs_strong, ATTENTION_SIZE,
                                                   self.sess_mask_strong,
                                                   softmax_stag=1, stag='1_2', mode='SUM', return_alphas=True)
            final_state_strong = tf.squeeze(final_state_strong, axis=1)
            # final_state_strong_query = final_state_strong
            final_state_strong_query = tf.concat([self.item_eb, final_state_strong], axis=-1)
            att_outputs_weak = kg_din_fcn_attention(final_state_strong_query, self.mid_sess_his_eb_weak, ATTENTION_SIZE,
                                                    self.sess_x_mask_weak,
                                                    softmax_stag=1, stag='1_3', mode='SUM', return_alphas=False)
            final_state_weak,self.weak_alpha = din_fcn_attention(self.item_eb, att_outputs_weak, ATTENTION_SIZE,
                                                 self.sess_mask_weak,
                                                 softmax_stag=1, stag='1_4', mode='SUM', return_alphas=True)
            # final_state_weak = mhsa_inter_triangle_aggregation(final_state_weak,
            #                                                      num_units=EMBEDDING_DIM * 2,
            #                                                      num_heads=4,
            #                                                      dropout_rate=0.,
            #                                                      is_training=True,name="weak")
            final_state_weak = tf.squeeze(final_state_weak, axis=1)
            # final_state_weak_query = final_state_weak
            # final_state_weak_query = tf.concat([self.item_eb, final_state_weak], axis=-1)

            att_outputs_general_cf = kg_din_fcn_attention(self.item_eb, self.mid_sess_his_eb_general_cf,
                                                          ATTENTION_SIZE,
                                                          self.sess_x_mask_general_cf,
                                                          softmax_stag=1, stag='1_12', mode='SUM', return_alphas=False)
            att_outputs_general_cf = mhsa_inter_triangle_aggregation(att_outputs_general_cf,
                                                                     num_units=EMBEDDING_DIM * 2,
                                                                     num_heads=4,
                                                                     dropout_rate=0.,
                                                                     is_training=True, name="general_cf")
            final_state_general_cf,self.cf_alpha = din_fcn_attention(self.item_eb, att_outputs_general_cf, ATTENTION_SIZE,
                                                       self.sess_mask_general_cf,
                                                       softmax_stag=1, stag='1_13', mode='SUM', return_alphas=True)
            final_state_general_cf = tf.squeeze(final_state_general_cf, axis=1)

            att_outputs_general = kg_din_fcn_attention(self.item_eb, self.mid_sess_his_eb_general,
                                                       ATTENTION_SIZE,
                                                       self.sess_x_mask_general,
                                                       softmax_stag=1, stag='1_5', mode='SUM', return_alphas=False)
            att_outputs_general = mhsa_inter_triangle_aggregation(att_outputs_general,
                                                                  num_units=EMBEDDING_DIM * 2,
                                                                  num_heads=4,
                                                                  dropout_rate=0.,
                                                                  is_training=True, name="general")
            final_state_general,self.kg_alpha = din_fcn_attention(self.item_eb, att_outputs_general, ATTENTION_SIZE,
                                                    self.sess_mask_general,
                                                    softmax_stag=1, stag='1_6', mode='SUM', return_alphas=True)
            final_state_general = tf.squeeze(final_state_general, axis=1)
        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state_strong, final_state_weak, final_state_general_cf,
             final_state_general],
            1)
        self.build_fcn_net(inp, use_dice=True, cl_emb=[final_state_general_cf, final_state_general])
