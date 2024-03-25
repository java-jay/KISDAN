# coding=utf-8
import numpy as np
import collections
import time
import pandas as pd
import itertools
import pickle as pkl
import pickle
import random
import networkx as nx


def split_data(ratings_final):
    rating_np = np.loadtxt(ratings_final, dtype=np.int32)
    test_indices = []
    user_count = {}  # 交互个数
    for user in np.unique(rating_np[:, 0]):
        user_data = rating_np[rating_np[:, 0] == user]
        if len(user_data) <= 4:
            continue
        user_count[user] = len(user_data)
        last_index_1 = np.where((rating_np == user_data[user_data[:, 2] == 1][-1]).all(axis=1))[0][-1]
        last_index_0 = np.where((rating_np == user_data[user_data[:, 2] == 0][-1]).all(axis=1))[0][-1]
        test_indices.append(last_index_1)
        test_indices.append(last_index_0)
    train_indices = np.setdiff1d(np.arange(rating_np.shape[0]), test_indices)
    train_data = rating_np[train_indices]
    test_data = rating_np[test_indices]
    test_users = test_data[:, 0]
    mask = np.isin(train_data[:, 0], test_users)
    train_data = train_data[mask]
    print(len(train_data))
    print(len(test_data))
    print(test_data[test_data[:, 0] == 2])
    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)


def get_cluster(max_item_id):
    start_time = time.time()
    file_name = "kg_final.txt"
    kg_np = np.loadtxt(file_name, dtype=np.int32)
    kg_np = np.unique(kg_np, axis=0)
    entity_counts = {}  # entity id出现的次数
    entity_items = {}
    item_entity = {}
    for head, relation, tail in kg_np:
        # if tail >= max_item_id and head < max_item_id:
        if tail >= max_item_id:
            if tail not in entity_counts:
                entity_counts[tail] = 0
            entity_counts[tail] += 1
        # if head >= max_item_id and tail < max_item_id:
        if head >= max_item_id:
            if head not in entity_counts:
                entity_counts[head] = 0
            entity_counts[head] += 1
    entity_counts = dict((key, value) for key, value in entity_counts.items() if value < 100000)
    for head, relation, tail in kg_np:
        if tail in entity_counts and head < max_item_id:
            if tail not in entity_items:
                entity_items[tail] = []
            if head not in entity_items[tail]:
                entity_items[tail].append(head)
            if head not in item_entity:
                item_entity[head] = []
            if tail not in item_entity[head]:
                item_entity[head].append(tail)
        if head in entity_counts and tail < max_item_id:
            if head not in entity_items:
                entity_items[head] = []
            if tail not in entity_items[head]:
                entity_items[head].append(tail)
            if tail not in item_entity:
                item_entity[tail] = []
            if head not in item_entity[tail]:
                item_entity[tail].append(head)
    print(len(entity_items))
    print(len(item_entity))
    df = pd.DataFrame()
    tmp_list = []
    for item_id in item_entity:
        for entity_id in item_entity[item_id]:
            tmp_list.append([item_id, entity_id, entity_items[entity_id]])
        print(item_id)
    colu = ["item_id", "entity_id", "neighbor_item_id"]
    df = pd.DataFrame(data=tmp_list, columns=colu)
    df['neighbor_item_id'] = df['neighbor_item_id'].astype('str')
    df = df.drop_duplicates('neighbor_item_id')
    df2 = pd.DataFrame()
    for index, row in df.iterrows():
        r = eval(row['neighbor_item_id'])
        if (len(r) <= 1):
            continue
        df2 = df2.append(row)
    df2['item_id'] = df['item_id'].astype('int')
    df2['entity_id'] = df['entity_id'].astype('int')
    df2.to_csv("kg_neighbor_item_id_1_hop.csv")
    print("耗时：" + str(time.time() - start_time) + "秒")


def train_cluster(flag):
    start_time = time.time()
    df = pd.read_csv("kg_neighbor_item_id_1_hop.csv")
    train_data = np.load("train_data.npy")
    # print train_data[train_data[:, 0] == 1]
    res = []
    columns = ["user_id", "user_cluster_item_id"]
    lis = df["neighbor_item_id"].tolist()
    lis2 = [eval(li) for li in lis]
    f2_dict = {}
    for user, item, target in train_data:
        if target == 1:
            f2_dict.setdefault(user, list()).append(item)
    for user in np.unique(train_data[:, 0]):
        behaviors = f2_dict.get(user, list())  # 用户看过的
        behaviors_cluster = set()  # 用户看过的且被聚类的
        if flag == "train":
            path = "train_cluster_1_hop_train.csv"
            behaviors = behaviors[:-1]  # 去除target
        else:
            path = "train_cluster_1_hop_test.csv"
        behaviors = set(behaviors)
        seen = set()
        for li in lis2:
            # 属于同一个entity id的item id聚类
            tmp = behaviors & set(li)
            if len(tmp) > 1:
                behaviors_cluster |= tmp
                tmp = list(tmp)
                if (user, frozenset(tmp)) not in seen:
                    res.append([user, tmp])
                    seen.add((user, frozenset(tmp)))
        for ff in behaviors - behaviors_cluster:
            res.append([user, [ff]])
        if (int(time.time() - start_time)) % 60 == 0:
            print(user)
    res_df = pd.DataFrame(data=res, columns=columns)
    res_df.to_csv(path)
    print("耗时：" + str(time.time() - start_time) + "秒")


def build_cf(flag):
    G = nx.Graph()
    triples = []
    edge_counts = {}
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    for user in np.unique(train_data[:, 0]):
        # print (user)
        user_data = train_data[train_data[:, 0] == user]
        f2 = []
        for row in user_data:
            if row[2] == 1:
                f2.append(int(row[1]))
        user_data = test_data[test_data[:, 0] == user]
        for row in user_data:
            if row[2] == 1:
                f2.append(int(row[1]))
        if flag == "train":
            f2 = f2[:-2]
            path = "cf_train.gexf"
        else:
            f2 = f2[:-1]
            path = "cf_test.gexf"
        for pair in itertools.combinations(f2, 2):
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
    # edge_counts = sorted(edge_counts.items(), key=lambda x: x[1], reverse=False)[-20:]
    # print edge_counts
    edge_counts = {key: value for key, value in edge_counts.items() if value >= 5}
    print(len(edge_counts))
    for key, value in edge_counts.items():
        G.add_edge(key[0], key[1])
    nx.write_gexf(G, path)
    print("写入完成")


def build_kg(max_item_id, path):
    G = nx.Graph()
    triples = []
    entity_counts = {}
    with open('kg_final.txt', 'r') as f:
        for line in f:
            triple = tuple(map(int, line.strip().split()))
            head = triple[0]
            tail = triple[2]
            if tail >= max_item_id and head < max_item_id:
                if tail not in entity_counts:
                    entity_counts[tail] = 0
                entity_counts[tail] += 1
            if head >= max_item_id and tail < max_item_id:
                if head not in entity_counts:
                    entity_counts[head] = 0
                entity_counts[head] += 1
    entity_counts = dict((key, value) for key, value in entity_counts.items() if value < 100)
    print(len(entity_counts))
    with open('kg_final.txt', 'r') as f:
        for line in f:
            triple = tuple(map(int, line.strip().split()))
            head = triple[0]
            tail = triple[2]
            if (head >= max_item_id and head not in entity_counts) or (
                    tail >= max_item_id and tail not in entity_counts):
                continue
            triples.append(triple)
    for triple in triples:
        G.add_edge(triple[0], triple[2], relation=triple[1])
    nx.write_gexf(G, path)
    print("写入完成")

def get_shortest_path_nodes(G, node1, node2):
    # 计算两个节点之间的最短路径，并返回该路径上的所有节点
    #
    # Parameters:
    # -----------
    # G : networkx.Graph
    #     图对象
    # node1 : int
    #     第一个节点的标识符
    # node2 : int
    #     第二个节点的标识符
    #
    # Returns:
    # --------
    # shortest_path_nodes : list
    #     最短路径经过的节点
    if G.has_node(node1) and G.has_node(node2):
        if not nx.has_path(G, node1, node2):
            return None
        shortest_path = nx.shortest_path(G, node1, node2)[1:-1]
        return shortest_path
    return None


def get_lst(lst):
    lst = [tuple(i) for i in lst]  # 转换为元组
    lst = list(set(lst))  # 去重
    lst = [list(i) for i in lst]  # 转换回列表
    return lst


def get_user_cluster(flag, max_item_id):
    if flag == "train":
        train_cluster = pd.read_csv("train_cluster_1_hop_train.csv")
        G = nx.read_gexf("cf_train.gexf")
        path = 'cluster_enhance_1_hop_cf_kg_train.pickle'
    else:
        G = nx.read_gexf("cf_test.gexf")
        train_cluster = pd.read_csv("train_cluster_1_hop_test.csv")
        path = 'cluster_enhance_1_hop_cf_kg_test.pickle'
    cluster_1_hop_dict = {}  # 知识图谱聚合
    for index, row in train_cluster.iterrows():
        uid = row["user_id"]
        if uid not in cluster_1_hop_dict:
            cluster_1_hop_dict[uid] = {}
            cluster_1_hop_dict[uid]["strong"] = []
            cluster_1_hop_dict[uid]["weak"] = []
            cluster_1_hop_dict[uid]["general_kg"] = []
            cluster_1_hop_dict[uid]["general_cf"] = []
        user_cluster_item_id = eval(row["user_cluster_item_id"])
        if isinstance(user_cluster_item_id, list):
            if len(user_cluster_item_id) == 1:
                cluster_1_hop_dict[uid]["weak"].append(user_cluster_item_id)
            else:
                cluster_1_hop_dict[uid]["strong"].append(user_cluster_item_id)
        else:
            cluster_1_hop_dict[uid]["weak"].append([user_cluster_item_id])
    print("kg.gexf")
    G2 = nx.read_gexf("kg.gexf")
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    f2_dict = {}
    for user, item, target in train_data:
        if target == 1:
            f2_dict.setdefault(user, list()).append(item)
    for user, item, target in test_data:
        if target == 1:
            f2_dict.setdefault(user, list()).append(item)
    count = 0
    print(len(f2_dict))
    for uid, f2 in f2_dict.items():
        if count % 1000 == 0:
            print(count)
        count += 1
        if flag == "train":
            target = f2[-2]
            f2 = f2[:-2]
        else:
            target = f2[-1]
            f2 = f2[:-1]
        for ff in f2:
            nodes = get_shortest_path_nodes(G, str(ff), str(target))
            if nodes is not None and len(nodes) > 0:
                int_nodes = [int(node) for node in nodes]
                cluster_1_hop_dict[uid]["general_cf"].append(int_nodes)
        for ff in f2:
            nodes = get_shortest_path_nodes(G2, str(ff), str(target))
            if nodes is not None and len(nodes) > 0:
                int_nodes = [int(node) for node in nodes if int(node) < max_item_id]
                cluster_1_hop_dict[uid]["general_kg"].append(int_nodes)
    print("get_lst")
    for uid in cluster_1_hop_dict:
        cluster_1_hop_dict[uid]["strong"] = get_lst(cluster_1_hop_dict[uid]["strong"])
        cluster_1_hop_dict[uid]["weak"] = get_lst(cluster_1_hop_dict[uid]["weak"])
        cluster_1_hop_dict[uid]["general_kg"] = get_lst(cluster_1_hop_dict[uid]["general_kg"])
        cluster_1_hop_dict[uid]["general_cf"] = get_lst(cluster_1_hop_dict[uid]["general_cf"])

    print(len(cluster_1_hop_dict))
    # 将字典对象保存到文件
    with open(path, 'wb') as f:
        pickle.dump(cluster_1_hop_dict, f)

split_data("ratings_final.txt")
get_cluster(24915)
train_cluster("train")
train_cluster("test")
build_kg(24915, "kg.gexf")
build_cf("train")
build_cf("test")
get_user_cluster("train", 24915)
get_user_cluster("test", 24915)
