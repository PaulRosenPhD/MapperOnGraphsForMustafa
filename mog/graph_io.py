from re import T
import networkx as nx
import json
import numpy as np


def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return [js_graph, nx.readwrite.node_link_graph(js_graph, directed=False, multigraph=False)]


def read_graph_file(filename):
    f = open(filename)
    graph = nx.Graph()
    for _x in f:
        x = _x.split()
        if x[0] == 'n':
            graph.add_node(x[1])
        if x[0] == 'e':
            graph.add_edge(x[1], x[2], value=1)
    return [None, graph]


def read_tsv_graph_file(filename):
    f = open(filename)
    graph = nx.Graph()
    for _x in f:
        if _x[0] == '#': continue
        x = _x.split()
        if not graph.has_node(x[0]):
            graph.add_node(x[0])
        if not graph.has_node(x[1]):
            graph.add_node(x[1])
        if graph.has_edge(x[0], x[1]):
            print("edge exists")
        if len(x) == 2:
            graph.add_edge(x[0], x[1], value=1)
        else:
            graph.add_edge(x[0], x[1], value=float(x[2]))
    return [None, graph]


def read_numpy_graphs(filename):
    data = np.load(filename, allow_pickle=True)
    # for key in data.keys():
    #     print("   variable name:", key          , end="  ")
    #     print("type: "+ str(data[key].dtype) , end="  ")
    #     print("shape:"+ str(data[key].shape))    

    X_train = data['tr_feat'] # node features
    A_train = list(data['tr_adj']) # list of adjacency matrices
    y_train = data['tr_class'] # class labels

    X_val = data['val_feat'] # node features
    A_val = list(data['val_adj']) # list of adjacency matrices
    y_val = data['val_class'] # class labels

    X_test = data['te_feat'] # node features
    A_test = list(data['te_adj']) # list of adjacency matrices
    y_test = data['te_class'] # class labels

    G_train = []
    for a, x in zip(A_train, X_train):
        G = nx.from_scipy_sparse_matrix(a)
        x_tuple = tuple(map(tuple, x))
        nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
        G_train.append(G)
        
    G_val = []
    for a, x in zip(A_val, X_val):
        G = nx.from_scipy_sparse_matrix(a)
        x_tuple = tuple(map(tuple, x))
        nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
        G_val.append(G)
        
    G_test = []
    for a, x in zip(A_test, X_test):
        G = nx.from_scipy_sparse_matrix(a)
        x_tuple = tuple(map(tuple, x))
        nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
        G_test.append(G)    

    return None, {'train': G_train, 'val': G_val, 'test': G_test}
    

def read_obj_graph(filename):
    vertices = []
    faces = []
    graph = nx.Graph()

    with open(filename, 'r') as f:
        for l in f.readlines():
            if l[0] == '#': continue
            elif len(l) == 1: continue
            elif l[0] == 'v': 
                vertices.append( l.split()[1:4] )
                graph.add_node( '#' + str(len(vertices)))
            elif l[0] == 'f': faces.append( l.split()[1:4] )
            else: print( l )

    edges = []
    for f in faces:
        fp = [ int(f[0]), int(f[1]), int(f[2]) ]
        edges.append( [fp[0], fp[1]] if fp[0] < fp[1] else [fp[1], fp[0]] )
        edges.append( [fp[1], fp[2]] if fp[1] < fp[2] else [fp[2], fp[1]]  )
        edges.append( [fp[2], fp[0]] if fp[2] < fp[0] else [fp[0], fp[2]]  )
    edges.sort()
    edges_unique = [edges[0]]
    for i in range(1, len(edges)):
        if edges[i][0] != edges[i-1][0] or edges[i][1] != edges[i-1][1]:
            edges_unique.append(edges[i])
            graph.add_edge( '#' + str(edges[i][0]), '#' + str(edges[i][1]), value=1)

    return [vertices,edges_unique,faces], graph


def read_filter_function(filename, ranked=False):
    with open(filename) as json_file:
        ff = json.load(json_file)

    if ranked:
        return ff['ranked']
    return ff['data']


def round_floats(o):
    if isinstance(o, float): return round(o, 4)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def _to_json(python_object) :
    if isinstance(python_object, np.float32):
        return float(python_object)
    elif isinstance(python_object, np.float64):
        return float(python_object)
    elif isinstance(python_object, np.int32):
        return int(python_object)
    elif isinstance(python_object, tuple) :
        python_object = {'__class__': 'tuple',
                         '__value__': list(python_object)}
    else :
        raise TypeError(str(type(python_object)) + ' is not JSON serializable') 

    return python_object


def write_json_data(filename, data):
    with open(filename, 'w') as outfile:
        #json.dump(round_floats(data), outfile, separators=(',', ':'))
        json.dump(data, outfile, separators=(',', ':'), default=_to_json)


def write_json_graph(filename, graph):
    write_json_data(filename, nx.node_link_data(graph))
