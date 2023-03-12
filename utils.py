import numpy as np
import numpy.ma as ma
from operator import or_

import numpy as np
import random
import matplotlib.pyplot as plt
#from sklearn.cluster.bicluster import SpectralCoclustering
import numpy.ma as ma
from operator import or_
import operator
import math
from numpy import genfromtxt
import random
from scipy.stats import pearsonr, spearmanr
#import distance_correlation as dc
from sklearn.datasets import make_biclusters
#from sklearn.datasets import samples_generator as sg
#from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from sklearn.metrics import consensus_score
import copy
from statistics import mean
# #polo
# from polo import optimal_leaf_ordering
# from scipy.spatial.distance import pdist
# from fastcluster import linkage
# from scipy.cluster.hierarchy import leaves_list
# from sklearn.metrics import mean_squared_error
np.random.seed(0)
random.seed(0)

# convert times
def convert_to_appropriate(arr): # iteration time to appropriate total time
    return [np.sum(arr[:i])-arr[0] for i in range(1, arr.shape[0]+1)]

def convert_times2(times, errors, threshold=300, intervals=30):
    offset = threshold/intervals
    # 0.1 - 0.2 - 0.3 - .... - threshold: step is 0.1
    times_disc = np.arange(0, threshold + offset, offset)
    new_errors = np.array([])
    i, j = 0, 0
    #print(len(times))
    n_times, n_times_disc = len(times), len(times_disc)
    while j < n_times: 
        while i < n_times_disc: 
            if times_disc[i] == times[j]:
                new_errors = np.append(new_errors, np.array(errors[j]))
                i += 1
                continue
            elif times_disc[i] < times[j]: 
                new_errors = np.append(new_errors, np.array(errors[j-1]))
                i += 1
                continue
            break
        j += 1
    diff = times_disc.shape[0] - new_errors.shape[0]
    for _ in range(diff):
        new_errors = np.append(new_errors, errors[-1])
    return times_disc.tolist(), new_errors.tolist()

def convert_times(times, errors, threshold=300, intervals=30):
    new_times, new_errors = [], []
    for t, e in zip(times, errors):
        new_t, new_e = convert_times2(t, e, threshold, intervals)
        new_times.append(new_t)
        new_errors.append(new_e)
    return new_times, new_errors


def plot_approx_pred_ten(error_trop, error_nmf, corr_trop, corr_nmf, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='upper right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.legend(['STMF', 'NMF'], loc='upper left')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def polo_clustering(data_param):
    data = copy.deepcopy(data_param)
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order = leaves_list(optimal_Z)
    data = data[opt_order]
    
    data = data.T  # transpose
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order_columns = leaves_list(optimal_Z)
    data = data[opt_order_columns]
    return data.T, opt_order, opt_order_columns

def rmse(X_orig, X_with_missing_values, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] == missing_value:
                #error = (abs(X_orig[i, j] - approx[i, j]))
                error = (X_orig[i, j] - approx[i, j])**2 
                errors.append(error)
    return np.sqrt(sum(errors)/len(errors))

def rmse_approx(X_orig, X_with_missing_values, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] != missing_value:
                #error = (abs(X_orig[i, j] - approx[i, j]))
                error = (X_orig[i, j] - approx[i, j])**2 
                errors.append(error)
    return np.sqrt(sum(errors)/len(errors))

def create_matrix_with_missing_values(X, percentage, missing_value):
    rows = X.shape[0]
    columns = X.shape[1]
    elements = rows * columns
    zero_elements = int(percentage * elements)
    for i in range(zero_elements):
        random_row = random.randint(0, rows-1)
        random_column = random.randint(0, columns-1)
        X[random_row, random_column] = missing_value
    return X

def check_zeros(X):
    rows = X.shape[0]
    columns = X.shape[1]
    for i in range(rows):
        for j in range(columns):
            if X[i, j] == 0:
                print("there is a zero element")
    return

def plot_approx_pred_ten(error_trop, error_nmf, corr_trop, corr_nmf, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='upper right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.legend(['STMF', 'NMF'], loc='upper left')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_corr_ten(error_trop, error_nmf, title, m, location, ylabel_1):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='lower right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return



def create_dict_five(X):
    n_lines = len(X)
    temp = 0
    dictionary = dict()
    for i in range(5, n_lines + 5, 5):
        list = np.asarray([line for line in X][i-5:i])
        dictionary[temp] = list
        temp += 1
    return dictionary

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
    
