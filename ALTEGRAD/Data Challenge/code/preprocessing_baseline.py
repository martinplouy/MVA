import os
import re
import random
import numpy as np
import networkx as nx
from time import time
import zipfile

# = = = = = = = = = = = = = = =

# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def random_walk(graph, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk


def random_walk_biased(graph, node, walk_length, bias_dict):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        neighbors_list = list(neighbors)
        probas = None
        if i > 0:
            probas = bias_dict[walk[i]]
        walk.append(np.random.choice(a=neighbors_list, p=probas))
    return walk


def generate_walks(graph, num_walks, min_walk_length, max_walk_length):
    '''
    samples num_walks walks of length between min_walk_length+1 and walk_length+1 from each node of graph
    '''
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    bias_dict = {}
    p = 1
    q = 0.5
    for n in graph_nodes:
        neighbors = graph.neighbors(n)
        neighbors_list = list(neighbors)
        probas = []
        for neigh in neighbors_list:
            shortest_path = nx.shortest_path_length(graph, n, neigh)
            if shortest_path == 0:
                probas.append(1/p)
            elif shortest_path == 2:
                probas.append(1/q)
            else:
                probas.append(1)

        Z = np.sum(probas)
        probas /= Z
        bias_dict[n] = probas
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk_length = np.random.randint(
                min_walk_length, max_walk_length + 1)
            walk = random_walk_biased(graph, nodes[j], walk_length, bias_dict)

            walk += [pad_vec_idx] * (max_walk_length + 1 - len(walk))
            walks.append(walk)

    return walks

# = = = = = = = = = = = = = = =


# 0-based index of the last row of the embedding matrix (for zero-padding)
pad_vec_idx = 1685894

# parameters
num_walks = 5
walk_length = 10
min_walk_length = 7
max_walk_length = 19
# maximum number of 'words' in each pseudo-document
max_doc_size = 100
path_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_to_data = path_root + '/data/'

# = = = = = = = = = = = = = = =
# this flag is meant to preprocess only a subsample of the data so that we can test quickly what works and what doesn't work
quick_preprocessing = False


def main():

    start_time = time()

    edgelists = os.listdir(path_to_data + 'edge_lists/')
    # important to maintain alignment with the targets!
    edgelists.sort(key=natural_keys)

    if quick_preprocessing:
        with open(path_to_data + 'train_idxs.txt', 'r') as file:
            train_idxs = file.read().splitlines()

        train_idxs = [int(elt) for elt in train_idxs]

        # # we only take graphs which are part of the training set
        edges_idxs = np.random.choice(train_idxs, 5000, replace=False)

        edgelists = [edgelists[elt] for elt in edges_idxs]

        with open(path_to_data + 'graphlists.txt', 'w') as f:
            for item in edgelists:
                itemArr = item.split('.')
                f.write("%s\n" % itemArr[0])

    docs = []
    for idx, edgelist in enumerate(edgelists):
        # construct graph from edgelist
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)

        # create the pseudo-document representation of the graph
        doc = generate_walks(g, num_walks, min_walk_length, max_walk_length)
        docs.append(doc)

        if len(edgelists) > 10 and idx % round(len(edgelists)/100) == 0:
            print(idx)

    print('documents generated')

    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    # 93000 docs have an average number of nodes of 18, and an average number of edges of 18.6
    docs = [d+[[pad_vec_idx]*(max_walk_length+1)]*(max_doc_size-len(d))
            if len(d) < max_doc_size else d[:max_doc_size] for d in docs]

    docs = np.array(docs).astype('int')
    print(docs.shape)
    print('document array shape:', docs.shape)

    np.save(path_to_data + 'documents.npy', docs, allow_pickle=False)

    print('documents saved')
    print('everything done in', round(time() - start_time, 2))

# = = = = = = = = = = = = = = =


if __name__ == '__main__':
    main()
