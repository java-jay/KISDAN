# coding=utf-8
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import os
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def get_input(n_samples, lengths_sess, mid_sess_1_hop, cat_sess_1_hop):
    maxlen_sess = numpy.max(lengths_sess)  # batch内最长的聚类数量
    tmp_list = [len(sublst) for lst2d in mid_sess_1_hop for sublst in lst2d]
    if len(tmp_list) == 0:
        maxlen_sess_x = 0
    else:
        maxlen_sess_x = max(tmp_list)  # 一个聚类中最大的交互数量
    for lst2d in mid_sess_1_hop:
        for sublst in lst2d:
            len(sublst)
    if maxlen_sess == 0:
        maxlen_sess = 1
    if maxlen_sess_x == 0:
        maxlen_sess_x = 1
    mid_sess_his = numpy.zeros((n_samples, maxlen_sess, maxlen_sess_x)).astype('int64')
    cat_sess_his = numpy.zeros((n_samples, maxlen_sess, maxlen_sess_x)).astype('int64')
    sess_mask = numpy.zeros((n_samples, maxlen_sess)).astype('float32')
    sess_x_mask = numpy.zeros((n_samples, maxlen_sess, maxlen_sess_x)).astype('float32')

    for idx, [sess_x, sess_y] in enumerate(
            zip(mid_sess_1_hop, cat_sess_1_hop)):
        for i, sublist in enumerate(sess_x):
            for j, val in enumerate(sublist):
                mid_sess_his[idx, i, j] = val
                sess_x_mask[idx, i, j] = 1.

        for i, sublist in enumerate(sess_y):
            for j, val in enumerate(sublist):
                cat_sess_his[idx, i, j] = val

        sess_mask[idx, :lengths_sess[idx]] = 1.
    return mid_sess_his, cat_sess_his, sess_mask, sess_x_mask


def prepare_data(input, target, maxlen=100, return_neg=False):
    # x: a list of sentences，如果maxlen不为空，后面会赋值，默认maxlen=100
    lengths_x = [len(s[4]) for s in input]  # 交互数量
    all_mid_sess_1_hop = [inp[7] + inp[8] for inp in input]
    all_cat_sess_1_hop = [inp[11] + inp[12] for inp in input]
    lengths_sess = [len(s) for s in all_mid_sess_1_hop]  # 聚类数量
    # lengths_sess = [len(s[7]) for s in input]  # 聚类数量
    lengths_sess_strong = [len(s[7]) for s in input]  # 强兴趣的聚类数量
    lengths_sess_weak = [len(s[8]) for s in input]  # 弱兴趣的聚类数量
    lengths_sess_general = [len(s[9]) for s in input]  # 潜在兴趣的聚类数量
    lengths_sess_general_cf = [len(s[10]) for s in input]  # 潜在兴趣的聚类数量
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    mid_sess_1_hop_strong = [inp[7] for inp in input]
    mid_sess_1_hop_weak = [inp[8] for inp in input]
    mid_sess_1_hop_general = [inp[9] for inp in input]
    mid_sess_1_hop_general_cf = [inp[10] for inp in input]
    cat_sess_1_hop_strong = [inp[11] for inp in input]
    cat_sess_1_hop_weak = [inp[12] for inp in input]
    cat_sess_1_hop_general = [inp[13] for inp in input]
    cat_sess_1_hop_general_cf = [inp[14] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])  # 取后maxlen个
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)  # batch size
    maxlen_x = numpy.max(lengths_x)  # batch内最大交互长度
    neg_samples = len(noclk_seqs_mid[0][0])  # 每个正样本对应5个负样本，neg_samples为5
    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(
            zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy
    # print ("lengths_x"+str(lengths_x))
    # print("mid_his"+str(mid_his.shape))
    # print ("noclk_mid_his"+str(noclk_mid_his.shape))
    # print ("mask" + str(mid_mask.shape))
    mid_sess_his_strong, cat_sess_his_strong, sess_mask_strong, sess_x_mask_strong = get_input(n_samples,
                                                                                               lengths_sess_strong,
                                                                                               mid_sess_1_hop_strong,
                                                                                               cat_sess_1_hop_strong)
    mid_sess_his_weak, cat_sess_his_weak, sess_mask_weak, sess_x_mask_weak = get_input(n_samples, lengths_sess_weak,
                                                                                       mid_sess_1_hop_weak,
                                                                                       cat_sess_1_hop_weak)
    mid_sess_his_general, cat_sess_his_general, sess_mask_general, sess_x_mask_general = get_input(n_samples,
                                                                                                   lengths_sess_general,
                                                                                                   mid_sess_1_hop_general,
                                                                                                   cat_sess_1_hop_general)
    mid_sess_his_general_cf, cat_sess_his_general_cf, sess_mask_general_cf, sess_x_mask_general_cf = get_input(
        n_samples,
        lengths_sess_general_cf,
        mid_sess_1_hop_general_cf,
        cat_sess_1_hop_general_cf)

    mid_sess_his, cat_sess_his, sess_mask, sess_x_mask = get_input(n_samples,
                                                                   lengths_sess,
                                                                   all_mid_sess_1_hop,
                                                                   all_cat_sess_1_hop)
    strong_to_weak = [mid_sess_his_strong, cat_sess_his_strong, sess_mask_strong, sess_x_mask_strong, mid_sess_his_weak,
                      cat_sess_his_weak, sess_mask_weak, sess_x_mask_weak, mid_sess_his_general, cat_sess_his_general,
                      sess_mask_general, sess_x_mask_general, mid_sess_his, cat_sess_his, sess_mask, sess_x_mask,
                      mid_sess_his_general_cf, cat_sess_his_general_cf, sess_mask_general_cf, sess_x_mask_general_cf]

    # print strong_to_weak[0]
    # print strong_to_weak[1]
    # print strong_to_weak[2]
    # print uids
    # print "strong"
    # print mid_sess_1_hop_strong
    # print mid_sess_his_strong
    # print "weak"
    # print mid_sess_1_hop_weak
    # print mid_sess_his_weak
    # print "general"
    # print mid_sess_1_hop_general
    # print mid_sess_his_general
    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(
            lengths_x), noclk_mid_his, noclk_cat_his, strong_to_weak

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), strong_to_weak


def eval(sess, test_data, model, model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    p_list = []
    t_list = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, strong_to_weak = prepare_data(
            src, tgt,
            return_neg=True)
        prob, loss, acc, aux_loss, strong_alpha, weak_alpha, kg_alpha, cf_alpha = model.calculate(sess,
                                                                                                  [uids, mids, cats,
                                                                                                   mid_his, cat_his,
                                                                                                   mid_mask, target, sl,
                                                                                                   noclk_mids,
                                                                                                   noclk_cats,
                                                                                                   strong_to_weak])

        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()

        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
            if p >= 0.5:
                p = 1
            else:
                p = 0
            p_list.append(p)
            t_list.append(t)

    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    f1 = f1_score(y_true=t_list, y_pred=p_list)
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, best_auc, f1


def train(
        train_file="local_train_splitByUser",
        test_file="local_test_splitByUser",
        uid_voc="uid_voc.pickle",
        mid_voc="mid_voc.pickle",
        cat_voc="cat_voc.pickle",
        batch_size=16,
        maxlen=100,
        test_iter=100,
        save_iter=100,
        model_type='DNN',
        seed=2,
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # train_data是返回的迭代操作符
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, "train", batch_size, maxlen,
                                  shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, "test", batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'KISDAN':
            model = Model_KISDAN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print(
            '                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- best_auc: %.4f ---- f1: %.4f' % eval(
                sess, test_data, model, best_model_path))
        sys.stdout.flush()
        sys.stdout.flush()
        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in range(3):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, strong_to_weak = prepare_data(
                    src, tgt,
                    maxlen=maxlen,
                    return_neg=True)
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr,
                                                         noclk_mids, noclk_cats, strong_to_weak])
                # uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, strong_to_weak = prepare_data(
                #     src, tgt,
                #     maxlen=maxlen,
                #     return_neg=False)
                # loss, acc,_ = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr,
                #                                strong_to_weak])
                loss_sum += loss
                accuracy_sum += acc
                # aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    if iter > 10000:
                        print(
                                '                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- best_auc: %.4f ---- f1: %.4f' % eval(
                            sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                # if iter > 100 and (iter % save_iter) == 0:
                #     print('save model iter: %d' % (iter))
                #     model.save(sess, model_path + "--" + str(iter))
            lr *= 0.5


def test(
        train_file="local_train_splitByUser",
        test_file="local_test_splitByUser",
        uid_voc="uid_voc.pickle",
        mid_voc="mid_voc.pickle",
        cat_voc="cat_voc.pickle",
        batch_size=16,
        maxlen=100,
        model_type='DNN',
        seed=2
):
    model_path = "dnn_best_model/ckpt_noshuffKGCTR34070.939741981061"
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, "train", batch_size, maxlen,
                                  shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, "test", batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'KISDAN':
            model = Model_KISDAN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        model.restore(sess, model_path)
        print(
                '                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- best_auc: %.4f ---- f1: %.4f' % eval(
            sess, test_data, model, model_path))


if __name__ == '__main__':
    # if len(sys.argv) == 4:
    #     SEED = int(sys.argv[3])
    # else:
    #     SEED = 3
    SEED = 3407
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    # train(model_type="KGCTR", seed=SEED)
    if sys.argv[1] == 'train':
        train(model_type=sys.argv[2], seed=SEED)
    elif sys.argv[1] == 'test':
        test(model_type=sys.argv[2], seed=SEED)
    # else:
    #     print('do nothing...')
