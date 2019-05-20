import tensorflow as tf
import numpy as np
from model import RippleNet
from topK import topK


def train(args, data_info):
    train_data = data_info[0]
    test_data = data_info[1]
    n_entity = data_info[2]
    n_relation = data_info[3]
    n_user = data_info[4]
    user_history_dict = data_info[5]
    ripple_set = data_info[7]

    model = RippleNet(args, n_entity, n_relation)

    user_embs = np.zeros((n_user,args.dim))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss, batch_user_embs = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))

                for i in range(batch_user_embs.shape[0]):
                    user_embs[train_data[start+i,0],:] = batch_user_embs[i,:]

                start += args.batch_size
                # print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            print('epoch %d    test auc: %.4f  acc: %.4f' % (step, test_auc, test_acc))

        saver.save(sess, "../model/model.ckpt")

        print("top-k recommendation evaluating ...")
        topK(train_data, test_data, user_embs, user_history_dict)


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop): 
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))

