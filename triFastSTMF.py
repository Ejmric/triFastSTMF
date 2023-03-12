import numpy as np
import numpy.ma as ma
from utils import max_plus, min_plus
import time
import copy
from collections import Counter
from misc import solve_A_X_B_C, three_max_plus

np.random.seed(0)

class triFastSTMF:
    """
    Fit a sparse tropical matrix factorization model for a matrix X.
    such that
        A = U V + E
    where
        A is of shape (m, n)    - data matrix
        U is of shape (m, rank) - approximated row space
        V is of shape (rank, n) - approximated column space
        E is of shape (m, n)    - residual (error) matrix
    """

    def __init__(self, rank_1=5, rank_2=5, criterion='convergence', max_iter=100, initialization='random_vcol',
                 epsilon=0.00000000000001, random_acol_param=5, threshold=300, seed_param=42, fixed_U=None, fixed_X=None, fixed_V=None):
        """
        :param rank: Rank of the matrices of the model.
        :param max_iter: Maximum nuber of iterations.
        """
        self.rank_1 = rank_1
        self.rank_2 = rank_2
        self.max_iter = max_iter
        self.initialization = initialization
        self.epsilon = epsilon
        self.random_acol_param = random_acol_param
        self.criterion = criterion  # convergence or iterations
        self.threshold = threshold
        self.is_transposed = False
        self.seed_param = seed_param
        self.fixed_U = copy.deepcopy(fixed_U)
        self.fixed_X = copy.deepcopy(fixed_X)
        self.fixed_V = copy.deepcopy(fixed_V)

    def b_norm(self, A):
        return np.sum(np.abs(A))

    def initialize_U(self, A, m, rank):
        U_initial = np.zeros((m, rank))
        k = self.random_acol_param  # number of columns to average
        if self.initialization == 'random':
            low = A.min()
            high = A.max()
            U_initial = low + (high - low) * np.random.rand(m, rank)  # uniform distribution
        elif self.initialization == 'random_vcol':
            # each column in U is element-wise average(mean) of a random subset of columns in A
            for s in range(rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].mean(axis=1)
        elif self.initialization == 'col_min':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].min(axis=1)
        elif self.initialization == 'col_max':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].max(axis=1)
        elif self.initialization == 'scaled':
            low = np.min(A)  # c
            high = 0
            U_initial = low + (high - low) * np.random.rand(m, rank)
        return ma.masked_array(U_initial, mask=np.zeros((m, rank)))
    
    def initialize_factor_matrix(self, A, rank):
        m = A.shape[0]
        initial_matrix = np.zeros((m, rank))
        k = self.random_acol_param  # number of columns to average
        if self.initialization == 'random':
            low = A.min()
            high = A.max()
            initial_matrix = low + (high - low) * np.random.rand(m, rank)  # uniform distribution
        elif self.initialization == 'random_vcol':
            # each column in factor matrix is element-wise average(mean) of a random subset of columns in A
            for s in range(rank):
                initial_matrix[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].mean(axis=1)
        elif self.initialization == 'col_min':
            for s in range(rank):
                initial_matrix[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].min(axis=1)
        elif self.initialization == 'col_max':
            for s in range(rank):
                initial_matrix[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].max(axis=1)
        elif self.initialization == 'scaled':
            low = np.min(A)  # c
            high = 0
            initial_matrix = low + (high - low) * np.random.rand(m, rank)
        return ma.masked_array(initial_matrix, mask=np.zeros((m, rank)))


    def assign_values(self, U, X, V, f, iterations, rows_perm, time, uvu, vuv, ulf, urf, errors, times):
        rows_perm_inverse = np.argsort(rows_perm)
        if self.is_transposed == True:
            self.U, self.X, self.V = V.T, X.T, U[rows_perm_inverse,:].T
        else:
            self.U, self.X, self.V = U[rows_perm_inverse,:], X, V
        self.error = f
        self.iterations = iterations
        self.time = time
        self.count_uvu, self.count_vuv = uvu, vuv
        self.count_ulf, self.count_urf = ulf, urf
        self.errors, self.times = errors, times

    def ULF(self, U, A, V, ind_j, ind_k, i): #F-ULF
        # UVU
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V
        U[i, ind_k] = A[i, ind_j] - V[ind_k, ind_j] # inplace change
        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))

        return U, V, U_old_k_vector, V_old_k_vector

    def URF(self, U, A, V, ind_j, ind_k, i): #F-URF
        # VUV
        U_old_k_vector = copy.deepcopy(U[:, ind_k])  # copying k-th column from U
        V_old_k_vector = copy.deepcopy(V[ind_k, :])  # copying k-th row from V
        V[ind_k, ind_j] = A[i, ind_j] - U[i, ind_k]  # inplace change
        mat_V = np.negative(V[ind_k, :]).reshape((-1, 1))
        U[:, ind_k] = min_plus(A, mat_V).reshape((-1))
        mat_U = np.negative(U[:, ind_k]).reshape((1, -1))
        V[ind_k, :] = min_plus(mat_U, A).reshape((-1))
        return U, V, U_old_k_vector, V_old_k_vector

    def return_old_U_and_V(self, U, V, U_old_k_vector, V_old_k_vector, ind_k):
        #print("returning old values")
        U[:, ind_k] = U_old_k_vector
        V[ind_k, :] = V_old_k_vector
        return

    def compute_td_element(self, i, j, A, U, V, k_list):
        mask, trop_k = A.mask, []
        for k in k_list:
            trop_k.append((U[i, k] + V[k, j]))
        max_k, max_k_elem_index = max(trop_k), np.argmax(trop_k)
        td_element = np.abs(A[i, j] - max_k)
        return td_element, max_k_elem_index

    def compute_td_row(self, i, A, U, V, k_list):
        m, n = A.shape  # mxn
        mask, td_row, j_list, td_row_k_indices = A.mask, 0, range(n), []
        for j in j_list:
            if not mask[i, j]:
                td_element, max_k_elem_index = self.compute_td_element(i, j, A, U, V, k_list)
                td_row += td_element
                td_row_k_indices.append((j, max_k_elem_index))
        return td_row, td_row_k_indices

    def compute_td_column(self, j, A, U, V, k_list):
        m, n = A.shape  # mxn
        mask, td_column, i_list, td_column_k_indices = A.mask, 0, range(m), []
        for i in i_list:
            if not mask[i, j]:
                td_element, max_k_elem_index = self.compute_td_element(i, j, A, U, V, k_list)
                td_column += td_element
                td_column_k_indices.append((j, max_k_elem_index))
        return td_column, td_column_k_indices

    def count_k(self, td_row_k_indices, td_column_k_indices):
        c_row = Counter([e[1] for e in td_row_k_indices])
        c_col = Counter([e[1] for e in td_column_k_indices])
        c_sum = c_row + c_col
        k_common = c_sum.most_common(1).pop()
        return k_common[0], k_common[1]
    
    def compute_factorization(self, f_old, f, f_new, iterations, basic_time, start_time, errors, times, U, V, A, fact_type="left-fact", current_X=None, factor_matrix=None):
        #print(i)
        m, n = A.shape
        mask = A.mask
        i_list, j_list = range(m), range(n)
        k_list = range(U.shape[1])  
        for i in i_list:
            #print(i)
            td_row, td_row_k_indices = self.compute_td_row(i, A, U, V, k_list)
            err, err_indices = [], []
            for j in j_list:
                if not mask[i, j]:
                    td_column, td_column_k_indices = self.compute_td_column(j, A, U, V, k_list)
                    err.append(td_column)
                    err_indices.append(td_column_k_indices)

            n_err = len(err)
            diffs_indices = sorted(range(n_err), key=lambda x: err[x], reverse=True)

            for index in range(n_err):  # finding the element which decreases error
                temp = diffs_indices[index]
                temp_j = err_indices[temp][0]
                ind_j = temp_j[0]
                ind_k, _ = self.count_k(td_row_k_indices, err_indices[temp])

                U, V, U_old_k_vector, V_old_k_vector = self.ULF(U, A, V, ind_j, ind_k, i)
                if fact_type=="left-fact":
                    f_new = self.b_norm(np.subtract(A, three_max_plus(U, current_X, factor_matrix)))
                else:
                    f_new = self.b_norm(np.subtract(A, three_max_plus(factor_matrix, current_X, V)))
                
                # save error and time for 1 iteration
                current_time = time.time() - start_time
                errors.append(f_new if f_new < f else f)
                times.append(round(current_time, 3))
                start_time = time.time()

                if f_new < f:
                    break
                self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind_k)
                if (time.time() - basic_time) >= self.threshold:
                    if fact_type=="left-fact":
                        self.assign_values(U, current_X, factor_matrix, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    else:
                        self.assign_values(factor_matrix, current_X, V, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return U, V, f_old, f, f_new, True, errors, times

                U, V, U_old_k_vector, V_old_k_vector = self.URF(U, A, V, ind_j, ind_k, i)
                if fact_type=="left-fact":
                    f_new = self.b_norm(np.subtract(A, three_max_plus(U, current_X, factor_matrix)))
                else:
                    f_new = self.b_norm(np.subtract(A, three_max_plus(factor_matrix, current_X, V)))
                    
                #urf += 1
                # save error and time for 1 iteration
                current_time = time.time() - start_time
                errors.append(f_new if f_new < f else f)
                times.append(round(current_time, 3))
                start_time = time.time()

                if f_new < f:
                    break
                self.return_old_U_and_V(U, V, U_old_k_vector, V_old_k_vector, ind_k)
                if (time.time() - basic_time) >= self.threshold:
                    if fact_type=="left-fact":
                        self.assign_values(U, current_X, factor_matrix, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    else:
                        self.assign_values(factor_matrix, current_X, V, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return U, V, f_old, f, f_new, True, errors, times

            if f_new < f:
                f_old, f = f, f_new
                if (time.time() - basic_time) >= self.threshold:
                    if fact_type=="left-fact":
                        self.assign_values(U, current_X, factor_matrix, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    else:
                        self.assign_values(factor_matrix, current_X, V, f, iterations, self.rows_perm, round(time.time() - start_time, 3), 0, 0,0,0, errors, times)
                    # save error and time for last iteration
                    current_time = time.time() - start_time
                    errors.append(f_new if f_new < f else f)
                    times.append(round(current_time, 3))
                    start_time = time.time()
                    return U, V, f_old, f, f_new, True, errors, times
            
        return U, V, f_old, f, f_new, False, errors, times
    
    def fit(self, A):
        """
        Fit model parameters U, V.
        :param A:
            Data matrix of shape (m, n)
            Unknown values are assumed to be masked.
        """
        basic_time = time.time()
        start_time = time.time()
        # check if matrix is wide
        m, n = A.shape  # rows, columns
        temp = False

        if m > n: # tall matrix
            self.is_transposed = True
            A = A.T # wide
            m, n = A.shape
            self.rank_1, self.rank_2 = self.rank_2, self.rank_1
            temp = True

        # permute matrix A, random rows
        rows_perm = np.random.RandomState(seed=self.seed_param).permutation(m) 
        self.rows_perm = rows_perm
        A = A[rows_perm, :]

        iterations = 0
        uvu, vuv = 0, 0
        ulf, urf = 0, 0
        errors, times = [], []

        if self.initialization=="fixed":
            if temp==True:
                U, X, V = self.fixed_V.T, self.fixed_X.T, self.fixed_U.T
                # for missing values set random value / masked values set to random value
                U[U==0] = np.random.choice(U[U!=0], size=len(U[U==0]))
                X[X==0] = np.random.choice(X[X!=0], size=len(X[X==0]))
                V[V==0] = np.random.choice(V[V!=0], size=len(V[V==0]))
            else:
                U, X, V = self.fixed_U, self.fixed_X, self.fixed_V
                U[U==0] = np.random.choice(U[U!=0], size=len(U[U==0]))
                X[X==0] = np.random.choice(X[X!=0], size=len(X[X==0]))
                V[V==0] = np.random.choice(V[V!=0], size=len(V[V==0]))
        else:
            U_initial = self.initialize_factor_matrix(A, self.rank_1)
            V_initial = self.initialize_factor_matrix(A.T, self.rank_2).T

            X = solve_A_X_B_C(U_initial, V_initial, A)
            U = min_plus(A, np.transpose(np.negative(max_plus(X, V_initial))))
            V = min_plus(np.transpose(np.negative(max_plus(U, X))), A)

        D = np.subtract(A, three_max_plus(U, X, V))

        # initialization of f values needed for convergence test
        norm = self.b_norm(D)
        f_old = norm + self.epsilon + 1
        f_new = norm
        f = f_new
        # save initial error
        current_time = time.time() - start_time
        errors.append(f)
        times.append(round(current_time, 3))
        start_time = time.time()

        while (f_old - f_new) > self.epsilon:
            f = f_new
            iterations += 1
            
            matrix_XV = max_plus(X,V) 
            U, XV, f_old, f, f_new, boolean_time, errors, times = self.compute_factorization(f_old, f, f_new, iterations, basic_time, start_time, errors, times, U, matrix_XV, A, "left-fact", X, V)
            
            if boolean_time:
                return
            
            matrix_UX = max_plus(U,X)
            UX, V, f_old, f, f_new, boolean_time, errors, times = self.compute_factorization(f_old, f, f_new, iterations, basic_time, start_time, errors, times, matrix_UX, V, A, "right-fact", X, U)
             
            if boolean_time:
                return
            
            X = solve_A_X_B_C(U, V, A)
            U = min_plus(A, np.transpose(np.negative(max_plus(X, V))))
            V = min_plus(np.transpose(np.negative(max_plus(U, X))), A)
            
            if (time.time() - basic_time) >= self.threshold:
                self.assign_values(U, X, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)
                return
                
                
        #print("triFastSTMF achieved the convergence by epsilon.")
        self.assign_values(U, X, V, f, iterations, rows_perm, round(time.time() - start_time, 3), uvu, vuv, ulf, urf, errors, times)


    def predict_all(self):
        """
        Return approximated matrix for all
        columns and rows.
        """
        return three_max_plus(self.U, self.X, self.V)

    def get_statistics(self, version, s, j, folder, transpose=False):
        results = [self.iterations, self.count_uvu, self.count_vuv, self.count_ulf, self.count_urf, self.time]
        if transpose == False:
            np.savetxt(folder + version + "/" + str(s) + "_" + str(j) + ".csv", results)
            np.savetxt(folder + version + "/errors/" + str(s) + "_" + str(j) + "_errors.csv", self.errors)
            np.savetxt(folder + version + "/times/" + str(s) + "_" + str(j) + "_times.csv", self.times)
        else:
            np.savetxt(folder + version + "/" + str(s)  + "_" + str(j) + "_transpose.csv", results)
            np.savetxt(folder + version + "/errors/" + str(s) + "_" + str(j) + "_errors_transpose.csv", self.errors)
            np.savetxt(folder + version + "/times/" + str(s) + "_" + str(j) + "_times_transpose.csv", self.times)
        return





