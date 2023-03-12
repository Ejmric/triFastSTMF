import numpy as np
import numpy.ma as ma
from operator import or_
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
import copy
from statistics import mean
import matplotlib.patches as mpatches
import itertools

def b_norm(A):
    return np.sum(np.abs(A))

def solve_system(U, data_matrix):
    V = min_plus(ma.transpose(np.negative(U)), data_matrix)
    return V

def random_acol_U(data_matrix, rank, m, k):
    factor_matrix = np.zeros((m, rank))
    for s in range(rank):
        factor_matrix[:, s] = data_matrix[:, np.random.randint(low=0, high=data_matrix.shape[1], size=k)].mean(axis=1)
    return ma.masked_array(factor_matrix, mask=np.zeros((m, rank)))

def get_coordinates(A):
    mask = A.mask
    rows = mask.shape[0]
    columns = mask.shape[1]
    coordinates = []
    for i in range(rows):
        for j in range(columns):
            if not mask[i, j]:
                coordinates.append((i, j))
    return coordinates

def get_max(product):
    data = product.data
    mask = product.mask
    rows = data.shape[0]
    columns = data.shape[1]
    result = []
    for j in range(columns):
        column_elements = []
        for i in range(rows):
            if mask[i, j] == False:
                column_elements.append(data[i, j])
        result.append(max(column_elements))
    return result

def get_min(product):
    data = product.data
    mask = product.mask
    rows = data.shape[0]
    columns = data.shape[1]
    result = []
    for j in range(columns):
        column_elements = []
        for i in range(rows):
            if not mask[i, j]:
                column_elements.append(data[i, j])
        if len(column_elements) == 0:  # only missing values
            raise Exception("there is an empty row/column in data")
        result.append(min(column_elements))
    return result

def max_plus(B, W):
    """
        :param B: numpy ndarray
        :param W: numpy ndarray
        :return:
        output: (max,+) multiplication of matrices B and W
        """
    rows_B, columns_B, columns_W = B.shape[0], B.shape[1], W.shape[1]
    B_size = np.size(B)
    W_size = np.size(W)
    
    if B_size * W_size != 0:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
        for i in range(0, rows_B):
            x = ma.expand_dims(ma.transpose(B[i, :]), axis=1)
            product = ma.array(x.data+W.data,mask=list(map(or_,x.mask,W.mask)))
            output[i, :]=get_max(product)
    else:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
    return output


def min_plus(B, W):
    """
        :param B: numpy ndarray
        :param W: numpy ndarray
        :return:
            output: (min,+) multiplication of matrices B and W
    """
    rows_B, columns_B, columns_W = B.shape[0], B.shape[1], W.shape[1]
    B_size = np.size(B)
    W_size = np.size(W)

    if B_size * W_size != 0:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
        for i in range(0, rows_B):
            x = ma.expand_dims(ma.transpose(B[i, :]), axis=1)
            product = ma.array(x.data+W.data,mask=list(map(or_,x.mask,W.mask)))
            output[i, :]=get_min(product)
    else:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
    return output

def solve_A_X_B_C(A, B, C):
    prod_A_C = min_plus(ma.transpose(np.negative(A)), C)
    X = min_plus(prod_A_C, ma.transpose(np.negative(B)))
    return X

def three_max_plus(U, X, V):
    res = max_plus(max_plus(U, X), V)
    return res


def construct_graph_with_missing_values(percentage=0.8, repeat=20):
    file = f'data/insecta-ant-colony3/final_graph.edges'
    with open(file, mode="r") as f:
        lines = f.readlines() 
        
    for i in range(0, repeat):
        #print(i)
        np.random.seed(42+i)
        missing_graph = np.random.choice(lines, size=int((1. - percentage)*len(lines)))
        new_graph = set(lines).difference(set(missing_graph))
    
        with open(f"data/insecta-ant-colony3/matrices/insecta-ant-colony3.newedges_{i}", mode="w") as f:
            for line in new_graph:
                f.write(line)
                
def compute_matrices(G):
    res_comm_size = []
    m, rank_1, rank_2, n = 0, 0, 0, 0
    for res in np.arange(0, 1.5, 0.05):
        communities = nx_comm.louvain_communities(G, seed=123, resolution=res, threshold=0.5)
        community_sizes = [len(community) for community in communities]
        res_comm_size.append((res, community_sizes, communities))
    four_partition_graphs = []
    for elem in res_comm_size:
        if len(elem[1])==4:
            four_partition_graphs.append(elem)
    X_indices, Y_indices, W_indices, Z_indices = [], [], [], []
    if len(four_partition_graphs) != 0:
        percentages_of_partitions = []
        for elem in four_partition_graphs:
            x, z, _, _ = sorted(elem[1], reverse=True)
            percentages_of_partitions.append((x+z)/sum(elem[1]))
        comm = np.argsort(percentages_of_partitions)[-1]
        communities = four_partition_graphs[comm][2]
        community_sizes = []
        for community in communities:
            #print(community)
            size = len(community)
            community_sizes.append(size)
        community_indices = np.argsort(community_sizes)
        rank_1, rank_2 = community_sizes[community_indices[1]], community_sizes[community_indices[0]]
        m, n = community_sizes[community_indices[3]], community_sizes[community_indices[2]]
        communities = list(communities)
        Y_partition, W_partition = communities[community_indices[1]],  communities[community_indices[0]]
        X_partition, Z_partition = communities[community_indices[3]],  communities[community_indices[2]]
        list_nodes = list(G.nodes())
        X_indices = [list_nodes.index(node_x) for node_x in X_partition]
        Y_indices = [list_nodes.index(node_y) for node_y in Y_partition]
        W_indices = [list_nodes.index(node_w) for node_w in W_partition]
        Z_indices = [list_nodes.index(node_z) for node_z in Z_partition]
        
        # Mapping dictionary
        mapping = {}
        for i, p in enumerate(X_partition):#, X_indices):
            mapping[p] = (i, "X")
        for i, p in enumerate(Y_partition):#, Y_indices):
            mapping[p] = (i, "Y")
        for i, p in enumerate(W_partition):#, W_indices):
            mapping[p] = (i, "W")
        for i, p in enumerate(Z_partition):#, Z_indices):
            mapping[p] = (i, "Z")
        
    indices = [m, rank_1, rank_2, n]
    return X_indices, Y_indices, W_indices, Z_indices, indices, mapping

def convert_sparse_to_dense(data, param="factor"):
    data = np.array(data.todense())
    missing_value = 0
    m, n = data.shape
    data = ma.masked_array(data, mask=np.zeros((m, n)))
    if param != "factor":
        data = ma.masked_equal(data, missing_value)
    return data


def get_matrices(adj, X_indices, Y_indices, W_indices, Z_indices):
    x = adj[X_indices]
    G_1 = x[:,Y_indices]
    ###
    x = adj[Y_indices]
    S = x[:,W_indices]
    ###
    x = adj[W_indices]
    G_2 = x[:,Z_indices]
    ###
    x = adj[X_indices]
    R = x[:, Z_indices]
    return R, G_1, S, G_2


def check_empty_rows_columns(R, G_1, G_2, rank_1, rank_2):
    zero_rows = np.where(np.all(R == 0, axis=1))
    #print(zero_rows)
    zero_columns = np.where(np.all(R == 0, axis=0))
    #print(zero_columns)
    R = np.delete(R, zero_rows, 0)
    R = np.delete(R, zero_columns, 1)
    m, n = R.shape
    mask_R = np.zeros((m, n))
    mask_R[R==0] = 1 # mask missing edges
    R = ma.masked_array(R, mask_R)
    G_1 = np.delete(G_1, zero_rows, 0)
    G_2 = np.delete(G_2, zero_columns, 1)
    # masks
    G_1 = ma.masked_array(G_1, np.zeros((m, rank_1)))
    G_2 =  ma.masked_array(G_2, np.zeros((rank_2, n)))
    return R, G_1, G_2



def return_graph(file, nodes):
    # Preprocess data types
    def process_line(l):
        temp = l.strip().split(" ")
        return (int(temp[0]), int(temp[1]), float(temp[2]))
    # Read edges from file
    with open(file, mode="r") as f:
        edges = [process_line(l) for l in f.readlines()]
    # Create empty graph
    g = nx.Graph()
    # Add node mapping same as in original graph
    g.add_nodes_from(nodes)
    # Add edges to the graphl, read from file
    g.add_weighted_edges_from(edges)
    return g


def convert_trifact_to_adj_R(N, G_1, S, G_2, G_mapping, F_mapping, fact="tropical"):
    A, B, C = None, None, None
    # Precompute all matrices
    if fact == "tropical":
        C = three_max_plus(G_1, S, G_2)
    else:
        C = ma.masked_array(np.dot(np.dot(G_1, S), G_2), mask=np.zeros((G_1.shape[0], G_2.shape[1])))
    # Empty matrix M
    M = ma.masked_array(np.zeros((N, N)), mask = np.ones((N, N)))
    # Convert trifactorization to adjacency matrix
    for n1 in G_mapping.keys():
        for n2 in G_mapping.keys():
            # i-index, p-partition, F_mapping - factor mapping
            (n1i, n1p), (n2i, n2p) = F_mapping[n1], F_mapping[n2]
            # indices of n1 and n2 nodes in G_mapping
            m1i, m2i = G_mapping[n1], G_mapping[n2]
            # If n1p == X && n2p == X then -1 ... etc.
            if n1p == "X" and n2p == "Z":
                M[m1i, m2i] = C[n1i, n2i]
                M.mask[m1i, m2i] = False
                M[m2i, m1i] = C[n1i, n2i] 
                M.mask[m2i, m1i] = False
            elif n1p == "Z" and n2p == "X":
                M[m1i, m2i] = C[n2i, n1i]
                M.mask[m1i, m2i] = False
                M[m2i, m1i] = C[n2i, n1i]
                M.mask[m2i, m1i] = False
    return M
    
def convert_trifact_to_adj(N, G_1, S, G_2, G_mapping, F_mapping, fact="tropical"):
    A, B, C = None, None, None
    # Precompute all matrices
    if fact == "tropical":
        A = max_plus(G_1, S)
        B = max_plus(S, G_2)
        C = three_max_plus(G_1, S, G_2)
    else:
        A = ma.masked_array(np.dot(G_1, S), mask=np.zeros((G_1.shape[0], S.shape[1])))
        B = ma.masked_array(np.dot(S, G_2), mask=np.zeros((S.shape[0], G_2.shape[1])))
        C = ma.masked_array(np.dot(np.dot(G_1, S), G_2), mask=np.zeros((G_1.shape[0], G_2.shape[1])))
    print(C.shape)
    # Empty matrix M
    M = np.zeros((N, N))
    # Convert trifactorization to adjacency matrix
    for n1 in G_mapping.keys():
        for n2 in G_mapping.keys():
            # i-index, p-partition, F_mapping - factor mapping
            (n1i, n1p), (n2i, n2p) = F_mapping[n1], F_mapping[n2]
            # indices of n1 and n2 nodes in G_mapping
            m1i, m2i = G_mapping[n1], G_mapping[n2]
            # If n1p == X && n2p == X then -1 ... etc.
            if n1p == "X" and n2p == "Y":
                M[m1i, m2i] = G_1[n1i, n2i]
                M[m2i, m1i] = G_1[n1i, n2i]
            elif n1p == "X" and n2p == "W":
                M[m1i, m2i] = A[n1i, n2i]
                M[m2i, m1i] = A[n1i, n2i]
            elif n1p == "X" and n2p == "Z":
                M[m1i, m2i] = C[n1i, n2i]
                M[m2i, m1i] = C[n1i, n2i]
            elif n1p == "Y" and n2p == "X":
                M[m1i, m2i] = G_1[n2i, n1i]
                M[m2i, m1i] = G_1[n2i, n1i]
            elif n1p == "Y" and n2p == "W":
                M[m1i, m2i] = S[n1i, n2i]
                M[m2i, m1i] = S[n1i, n2i]
            elif n1p == "Y" and n2p == "Z":
                M[m1i, m2i] = B[n1i, n2i]
                M[m2i, m1i] = B[n1i, n2i]
            elif n1p == "W" and n2p == "Z":
                M[m1i, m2i] = G_2[n1i, n2i]
                M[m2i, m1i] = G_2[n1i, n2i]
            elif n1p == "W" and n2p == "Y":
                M[m1i, m2i] = S[n2i, n1i]
                M[m2i, m1i] = S[n2i, n1i]
            elif n1p == "W" and n2p == "X":
                M[m1i, m2i] = A[n2i, n1i]
                M[m2i, m1i] = A[n2i, n1i]
            elif n1p == "Z" and n2p == "W":
                M[m1i, m2i] = G_2[n2i, n1i]
                M[m2i, m1i] = G_2[n2i, n1i]
            elif n1p == "Z" and n2p == "Y":
                M[m1i, m2i] = B[n2i, n1i]
                M[m2i, m1i] = B[n2i, n1i]    
            elif n1p == "Z" and n2p == "X":
                M[m1i, m2i] = C[n2i, n1i]
                M[m2i, m1i] = C[n2i, n1i]
    return M


def rmse(X_orig, approx):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    mask = X_orig.mask
    for i in range(rows):
        for j in range(columns):
            if mask[i, j] == False:
                #error = (abs(X_orig[i, j] - approx[i, j]))
                error = (X_orig[i, j] - approx[i, j])**2 
                errors.append(error)
    return np.sqrt(sum(errors)/len(errors))


def open_data(fname):
    with open(fname, mode="r") as f:
        lines = f.readlines()
        lines = [l.strip().split(",") for l in lines]
        headers = lines[0]
        data = []
        for l in lines[1:]:
            data.append([float(v) for v in l])
    return headers, data



def plot_days(param_data1, param_data2, param_data3):#, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    #fig.suptitle(title)
    cmap = "jet"

    min_a, max_a = np.min(param_data1), np.max(param_data1)
    min_d, max_d = np.min(param_data2), np.max(param_data2)
    min_e, max_e = np.min(param_data3), np.max(param_data3)
    # compute final min and max
    final_min = min([min_a, min_d, min_e])
    final_max = 30
    total_max = max([max_a, max_d, max_e])

    a = ax[0].pcolor(param_data1, vmin=final_min, vmax=final_max, cmap=cmap)
    ax[0].set_title("Days 1-19")
    ax[0].set_ylabel("Ant", fontsize=16)
    ax[0].set_xlabel("Ant", fontsize=16)
    ax[0].invert_yaxis()

    d = ax[1].pcolor(param_data2, vmin=final_min, vmax=final_max, cmap=cmap)
    ax[1].set_title("Days 20-31")
    ax[1].set_xlabel("Ant", fontsize=16)
    ax[1].set_ylabel("Ant", fontsize=16)
    ax[1].invert_yaxis()

    e = ax[2].pcolor(param_data3, vmin=final_min, vmax=final_max, cmap=cmap)
    ax[2].set_title("Days 32-41")
    ax[2].set_ylabel("Ant", fontsize=16)
    ax[2].set_xlabel("Ant", fontsize=16)
    ax[2].invert_yaxis()
    
   
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax, extend="max")#, fontsize=12)
    plt.text(0.36, 32.2, str(int(total_max)))
    plt.savefig("figures/cosine_adjacency_plots.png", bbox_inches = 'tight', dpi=600)
    plt.show();
    
def multilayered_graph(G_1, S, G_2, *subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()

    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)

    offset_x, offset_y = 0, 0
    for mat in [G_1, S, G_2]:
        x, y = np.nonzero(mat) 
        offset_y += mat.shape[0]
        G.add_edges_from(zip(list(x+offset_x), list(y+offset_y)), color="black")
        offset_x = offset_y
    
    return G

#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def compute_jaccard_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices):
    # m, rank_1, rank_2, n, nodes
    nodes = m + n + rank_1 + rank_2
    orig_X, orig_Y, orig_W, orig_Z = list(range(m)), list(range(m, m+rank_1)), list(range(m+rank_1, m+rank_1+rank_2)), list(range(m+rank_1+rank_2,nodes))
    X_js = jaccard(X_indices, orig_X)
    Y_js = jaccard(Y_indices, orig_Y)
  
    W_js = jaccard(W_indices, orig_W)
    Z_js = jaccard(Z_indices, orig_Z)
    js = mean([X_js, Y_js, W_js, Z_js])
    return js

def compute_adjusted_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices):
    # m, rank_1, rank_2, n, nodes
    nodes = m + n + rank_1 + rank_2
    labels_true = np.array([0]*m + [1]*rank_1 + [2]*rank_2 + [3]*n)
    labels_pred = np.array([0]*nodes) # initialization
    labels_pred[X_indices] = [0]*m
    labels_pred[Y_indices] = [1]*rank_1
    labels_pred[W_indices] = [2]*rank_2
    labels_pred[Z_indices] = [3]*n
    ars = adjusted_rand_score(labels_true, labels_pred)
    return ars

def compute_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices):
    # m, rank_1, rank_2, n, nodes
    nodes = m + n + rank_1 + rank_2
    labels_true = np.array([0]*m + [1]*rank_1 + [2]*rank_2 + [3]*n)
    labels_pred = np.array([0]*nodes) # initialization
    labels_pred[X_indices] = [0]*m
    labels_pred[Y_indices] = [1]*rank_1
    labels_pred[W_indices] = [2]*rank_2
    labels_pred[Z_indices] = [3]*n
    rs = rand_score(labels_true, labels_pred)
    return rs
    
def random_partitioning(m, rank_1, rank_2, n, adj, list_of_indices, seed, plot=False):
    np.random.seed(seed)
    lista = copy.deepcopy(list_of_indices)
    np.random.shuffle(lista)
    X_indices, Y_indices, W_indices, Z_indices = lista[:m], lista[m:m+rank_1], lista[m+rank_1:m+rank_1+rank_2], lista[m+rank_1+rank_2:]
    js = compute_jaccard_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices) 
    ars = compute_adjusted_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices)
    rs = compute_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices)
    adj = np.array(adj.todense())
    ###
    x = adj[X_indices]
    G_1 = x[:,Y_indices]
    ###
    x = adj[Y_indices]
    S = x[:,W_indices]
    ###
    x = adj[W_indices]
    G_2 = x[:,Z_indices]
    ###
    x = adj[X_indices]
    R = x[:, Z_indices]
    ###
    G_1 = ma.masked_array(G_1, np.zeros((m, rank_1)))
    S =  ma.masked_array(S, np.zeros((rank_1, rank_2)))
    G_2 =  ma.masked_array(G_2, np.zeros((rank_2, n)))
    mask_R = np.zeros((m, n))
    mask_R[R==0] = 1
    R = ma.masked_array(R, mask_R)
    labeldict = {i: v for i, v in enumerate(lista)}
    blue_patch = mpatches.Patch(color='#4169E1', label='Partition A from network K')
    pink_patch = mpatches.Patch(color='#E05882', label='Partition B from network K')
    purple_patch = mpatches.Patch(color='#A151E0', label='Partition C from network K')
    orange_patch = mpatches.Patch(color='#CE8346', label='Partition D from network K')

    subset_sizes = [m, rank_1, rank_2, n]
    list_colors = ["#4169E1"]*45 + ["#E05882"]*10 + ["#A151E0"]*15 + ["#CE8346"]*30
    subset_color = [
       "gold",
       "violet",
       "limegreen",
       "darkorange",
    ]

    G = multilayered_graph(G_1, S, G_2, *subset_sizes)
    color = []
    real_nodes = list(labeldict.values())
    for v in real_nodes:
        #print(G.nodes)
        if int(v) < 45:
            color.append("#4169E1")
        elif 45<=v<55:
            color.append("#E05882")
        elif 55<=v<70:
            color.append("#A151E0")
        else:
            color.append("#CE8346")

    pos = nx.multipartite_layout(G, subset_key="layer")

    x, y = np.nonzero(R) 
    w = np.shape(G_1)[0]+np.shape(S)[0]+np.shape(G_2)[0]
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    
    if plot:
        plt.figure(figsize=(12, 13))
        nx.draw(G, pos, node_color=color, edge_color=(0,0,0,0.5), font_color="white",  node_size=500, font_size=11) # with_labels=True, labels=labeldict
    
        plt.text(-0.0605,-1.08,"X", fontsize=16)
        plt.text(-0.0145,-0.28,"Y", fontsize=16)
        plt.text(0.030,-0.40,"W", fontsize=16)
        plt.text(0.076,-0.74,"Z", fontsize=16)
        #plt.legend(handles=[blue_patch, pink_patch, purple_patch, orange_patch], prop={"size":14})
        plt.savefig("figures/random_partitioning_" + str(seed) + ".png")
        plt.show();
    #print("jaccard score: " + str(js))
    #print("adjusted rand score: " + str(ars))
    #print("rand score: " + str(rs))
    return R, G_1, S, G_2, js, rs, ars

def pseudorandom_partitioning(m, rank_1, rank_2, n, adj, list_of_indices, seed, plot=False):
    np.random.seed(seed)
    lista = copy.deepcopy(list_of_indices)
    X_indices, Z_indices = lista[:m], lista[m+rank_1+rank_2:]
    two_partitions_indices = lista[m:m+rank_1+rank_2]
    np.random.shuffle(two_partitions_indices)
    Y_indices, W_indices = lista[m:m+rank_1], lista[m+rank_1:m+rank_1+rank_2]
    js = compute_jaccard_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices)
    ars = compute_adjusted_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices)
    rs = compute_rand_score(m, rank_1, rank_2, n, X_indices, Y_indices, W_indices, Z_indices)
    adj = np.array(adj.todense())
    ###
    x = adj[X_indices]
    G_1 = x[:,Y_indices]
    ###
    x = adj[Y_indices]
    S = x[:,W_indices]
    ###
    x = adj[W_indices]
    G_2 = x[:,Z_indices]
    ###
    x = adj[X_indices]
    R = x[:, Z_indices]
    ###
    G_1 = ma.masked_array(G_1, np.zeros((m, rank_1)))
    S =  ma.masked_array(S, np.zeros((rank_1, rank_2)))
    G_2 =  ma.masked_array(G_2, np.zeros((rank_2, n)))
    mask_R = np.zeros((m, n))
    mask_R[R==0] = 1
    R = ma.masked_array(R, mask_R)
    labeldict = {i: v for i, v in enumerate(lista)}
    blue_patch = mpatches.Patch(color='#4169E1', label='Partition A from network K')
    pink_patch = mpatches.Patch(color='#E05882', label='Partition B from network K')
    purple_patch = mpatches.Patch(color='#A151E0', label='Partition C from network K')
    orange_patch = mpatches.Patch(color='#CE8346', label='Partition D from network K')

    subset_sizes = [m, rank_1, rank_2, n]
    subset_color = [
       "gold",
       "violet",
       "limegreen",
       "darkorange",
    ]

    G = multilayered_graph(G_1, S, G_2, *subset_sizes)
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    x, y = np.nonzero(R) 
    w = np.shape(G_1)[0]+np.shape(S)[0]+np.shape(G_2)[0]
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    color = []
    real_nodes = list(labeldict.values())
    for v in real_nodes:
        #print(G.nodes)
        if int(v) < 45:
            color.append("#4169E1")
        elif 45<=v<55:
            color.append("#E05882")
        elif 55<=v<70:
            color.append("#A151E0")
        else:
            color.append("#CE8346")
    
    if plot:
        plt.figure(figsize=(12, 13))
        nx.draw(G, pos, node_color=color, edge_color=(0,0,0,0.5), font_color="white", node_size=500, font_size=11) #  with_labels=True, labels=labeldict
    
        plt.text(-0.0605,-1.08,"X", fontsize=16)
        plt.text(-0.0145,-0.28,"Y", fontsize=16)
        plt.text(0.030,-0.40,"W", fontsize=16)
        plt.text(0.076,-0.74,"Z", fontsize=16)
        #plt.legend(handles=[blue_patch, pink_patch, purple_patch, orange_patch], prop={"size":14})
        plt.savefig("figures/pseudorandom_partitioning_" + str(seed) + ".png")
        plt.show();
    #print("jaccard score: " + str(js))
    #print("adjusted rand score: " + str(ars))
    #print("rand score: " + str(rs))
    return R, G_1, S, G_2, js, rs, ars