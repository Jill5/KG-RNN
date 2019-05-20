import collections
import os
import numpy as np
import random

kg = collections.defaultdict(list)      #kg是一个dict，key是head，值是一个list，list的元素是以head为头结点的三元组的(tail,relation)

def load_data(args):

    #train_data, eval_data, test_data, user_history_dict = load_rating(args)
    train_data, test_data, user_history_dict, user_dislike_dict, n_user = load_rating(args)
    n_entity, n_relation = load_kg(args)
    ripple_set = get_ripple_set(args, user_history_dict)

    return train_data, test_data, n_entity, n_relation, n_user, user_history_dict, user_dislike_dict, ripple_set
    #return train_data, eval_data, test_data, n_entity, n_relation, ripple_set
    # return train_data, test_data, n_entity, n_relation, ripple_set

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # 随机划分rating数据集，train:test = 8:2
    #eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    # eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    # left = set(range(n_ratings)) - set(eval_indices)
    # test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    # train_indices = list(left - set(test_indices))

    test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(set(range(n_ratings)) - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    # user_history_dict是key为user的dict，值是一个list，list元素是train数据集中user喜欢的item
    n_user = 0
    user_history_dict = dict()
    user_dislike_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if user > n_user:
            n_user = user
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)
        else:
            if user not in user_dislike_dict:
                user_dislike_dict[user] = []
            user_dislike_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict] #只保留在user_history_dict里的user的rating
    #eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]   #同上
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]   #同上
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    #eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    n_user += 1

    #return train_data, eval_data, test_data, user_history_dict
    return train_data, test_data, user_history_dict, user_dislike_dict, n_user


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
	# 注意kg_final的format: h,r,t
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    construct_kg(kg_np)

    return n_entity, n_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))



def get_ripple_set(args, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    # 注意：ripple_set是一个dict，key是user，值是list，list元素是triple，如上，而triple的元素也是list，如hop_0_heads，元素是hop0里的所有head

    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                # indices = sorted(np.random.choice(len(memories_h), size=args.n_memory, replace=replace))
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set


def generate_negative_samples(n_entity):
    print('generating negative samples...')
    corrupted_tuples = list()   #corrupted tuple, format:(h, t, r, t_neg)

    for head in kg:
        for tail_and_relation in kg[head]:
            for i in range(3):
                tail_neg = random.randint(0,n_entity-1)
                while (tail_neg, tail_and_relation[1]) in kg[head]:
                    tail_neg = random.randint(0,n_entity-1)
                corrupted_tuples.append((head, tail_and_relation[0], tail_and_relation[1], tail_neg))

            # tail_neg_2 = random.randint(0,n_entity-1)
            # corrupted_tuples.append((tail_and_relation[0], head, tail_and_relation[1], tail_neg_2))

    return corrupted_tuples



