import random
from typing import Optional

import numpy as np
import pandas as pd

from modelling.debug import DEBUG_NSFM, DEBUG_RBF, DEBUG_PSO, DEBUG_PSO_SPARSE

"""
https://ieeexplore.ieee.org/document/6395832
"""


def subspace_input_distance(u_k, *, l, N: int, a: list[list[int]], delta_a: list[int]):  # d^l_r
    return sum([
        (a[l-1][i-1]-u_k[i-1])**2/(N*delta_a[i-1]**2) for i in range(1, N+1)
    ])**.5

def nsfm_algorithm(*, U_train, Y_train, s: np.ndarray) -> (int, list[list[int]]):
    """
    :param U_train:
    :param Y_train:
    :param s:
    :return:
        L: number of selected RBF centers
        U: selected RBF center locations
    """
    if DEBUG_NSFM: print(f"NSFM ALGORITHM >> PARTITION: {s}")

    N = len(s)
    L = 1  # prepare calculating the first RBF center (0-INDEXED)

    max_U_train = np.max(U_train, axis=0)
    min_U_train = np.min(U_train, axis=0)

    if DEBUG_NSFM: print(f"MAX U_train {max_U_train}")
    if DEBUG_NSFM: print(f"MIN U_train {min_U_train}")

    # s[i] fuzzy set partitioning of dim. i
    delta_a = [(max_U_train[i-1]-min_U_train[i-1]) / (s[i-1]-1) for i in range(1, N+1)]
    # fuzzy set centers (s[i]+1 for inclusive range)
    a = [[min_U_train[i-1]+delta_a[i-1]*j for j in range(0, s[i-1]+1)] for i in range(1, N+1)]
    if DEBUG_NSFM: print(f"FUZZY SET PARTITIONING >> DELTA A: <{delta_a}>")
    if DEBUG_NSFM: print(f"FUZZY SET PARTITIONING >> A: <{a}>")

    if DEBUG_NSFM: print(f"DATA POINT [1] >> TRAINING DATA POINT: <{U_train[0]}>")
    u_hat_1 = []  # generate the first RBF centers
    for i in range(1, N+1):  # 1 to N (1-indexed)
        # A^L_i, the center (partitioned by 1<=j<=s_i) with the max. membership for that dim. i & L
        # A_i,j to be the partitioning of each dim. i
        # we're using max instead of if/else <= 1, since d is always positive (refer to equation 9)
        # the equation is derived to be simpler by assuming the current dim. i
        # iterating each points of the partition (j over range(1,s_i))
        A_L_i = max([
            min(1-((a[i-1][j-1]-U_train[1-1][i-1])**2/(N*delta_a[i-1])**2)**0.5, 1)
            for j in range(1, s[i-1]+1)
        ])
        if DEBUG_NSFM: print(f"[0] FEATURE <{i}> >> CENTER: <{A_L_i}>")
        u_hat_1.append(A_L_i)
    if DEBUG_NSFM:  print(f"CENTER [1]: <{u_hat_1}>")

    u_hat = [u_hat_1]

    k: int  # data point
    for k in range(2, len(U_train)+1):  # for each data point (k=1 has been done by initialization on previous loop)
        # print(f"DATA POINT [{k}]")

        # if data point k lies outside the hyper-ellipses defined by the already selected center
        if min([subspace_input_distance(U_train[k-1], l=l, N=N, a=u_hat, delta_a=delta_a) for l in range(1, L+1)]) > 1:
            L += 1
            if DEBUG_NSFM: print(f"[{k}] NEW RBF CENTER GENERATED: <{L}>")

            u_hat_L = []  # generate the first RBF centers
            for i in range(1, N+1):  # 1 to N (1-indexed)
                A_L_i = max([
                    min(1-((a[i-1][j-1]-U_train[1-1][i-1])**2/(N*delta_a[i-1])**2)**0.5, 1)
                    for j in range(1, s[i-1]+1)
                ])
                u_hat_L.append(A_L_i)
            u_hat.append(u_hat_L)  # generate Lth RBF centers

            if DEBUG_NSFM: print(f"CENTER <{L}>: {u_hat_L}")

    return L, u_hat


class RBFNetwork:
    def __init__(self, *, u_hat: np.ndarray, L: int, N: int, w: Optional[np.array] = None):
        self.N = N
        self.L = L

        self.w: np.ndarray = np.ones((L,)) if w is None else w
        self.u_hat: np.ndarray = u_hat  # provided RBF centers

        self.mu_l = lambda u_k, u_hat_l: np.linalg.norm(u_k-u_hat_l)

        # g(mu)
        self.activation = lambda mu: (mu**2)*np.log(mu)


    def forward(self, u_k) -> np.ndarray:
        # hidden node response
        if DEBUG_RBF: print([
                self.u_hat[l-1]
                for l in range(1, self.L+1)
            ])
        self.z_k = np.ndarray(
            [
                self.activation(self.mu_l(u_k, self.u_hat[l-1]))
                for l in range(1, self.L+1)
            ]
        )
        return np.dot(self.z_k, self.w)

    def train(self, U_train, Y_train: np.ndarray):
        Z = np.array([])
        for u_k in U_train:
            self.forward(u_k)
            Z.apppend(self.z_k)
        Z_T = np.transpose(Z)

        self.w = (np.transpose(Y_train) @ Z_T) @ np.linalg.inv(Z @ Z_T)


def RMSE_criterion_fitness(actual: np.array, pred: np.array, *, N: int):
    return ((sum((actual-pred)**2)**0.5)/N)**0.5


def pso_algorithm(U_train: np.ndarray, Y_train: np.ndarray, U_val: np.ndarray, Y_val: np.ndarray, *,
                  s_min: int, s_max: int, P: int, c_1: float, c_2: float, V_max: int,
                  max_simulation_step: int, R_norm_threshold: float) -> (int, list[int], list[int]):
    """
    :param U_train:
    :param Y_train:
    :param U_val:
    :param Y_val:
    :param s_min: min # of fuzzy sets
    :param s_max: max # of fuzzy sets
    :param P: swarm population size
    :param c_1: PSO operational parameter, cognitive component
    :param c_2: PSO operational parameter, social component
    :param V_max: PSO operational parameter, velocity clamping
    :param max_simulation_step:
    :param R_norm_threshold:
    :return:
        L_f: selected amount of RBF centers
        U_f: selected RBF center locations
        w_f: selected RBF synaptic weights (for the last linear layer)
    """
    N = U_train.shape[1]  # feature dimension

    v: np.ndarray = np.zeros((P, N))  # vectors of INT vecs
    # vvvv also particle positions
    s: np.ndarray = np.random.randint(s_min, s_max+1, size=(P,N))  # populations of N-lengthed fuzzy set  (AKA x_ij(t))

    # following to lines (y_i and s_f) are tightly related
    # vvvv currently sets an initial position from the initial s_i values
    y_f = s.copy()  # personal best position y_i for each particle
    s_f = np.full((P,), np.inf)  # (initial) fitness value for each particle's personal best position y_i
    y_hat_f = y_f[0]  # global best position
    s_hat_f = s_f[0]  # global best position's fitness value
    t = 1

    f = lambda x: RMSE_criterion_fitness(Y_val, x, N=N)

    L = np.zeros((P,), dtype=int)  # list of number of centers in each RBF Network (particle)
    U_hat = [[[0 for _c in range(1, N+1)] for _u in range(1, L[l-1]+1)] for l in range(1, P+1)]  # list of center locations in each RBF Network (particle) [shape not rectangular]
    w = [np.ones((L[l-1],)) for l in range(1, P+1)]  # list of weights in each RBF Network (particle)
    L_f = L[0]
    U_hat_f = U_hat[0]
    w_f = w[0]

    R_max_1 = None
    end_loop = False

    while not end_loop:
        if DEBUG_PSO_SPARSE: print(f"PARTICLE SIMULATION STEP <{t}>")

        # create the RBF-NN network
        for i in range(1, P+1):
            if DEBUG_PSO: print(f"PARTICLE <{i}>")

            L[i], U_hat[i] = nsfm_algorithm(U_train=U_train, Y_train=Y_train, s=s[i])
            if DEBUG_PSO_SPARSE: print(f"-- NSFM RBF-NN GENERATION >> NUMBER OF CENTER: <{L[i]}> RBF CENTER LOCATION: {U_hat[i]}")

            # calculate the synaptic weights
            rbf = RBFNetwork(u_hat=U_hat[i], L=L[i], N=N)
            rbf.train(U_train=U_train, Y_train=Y_train)
            w[i] = rbf.w

            # calculate fitness function and update personal/global best positions if needed
            U_pred = np.array([])
            for u_k in U_val:
                U_pred.append(rbf.forward(u_k), axis=0)

            f_x_i = f(U_pred)
            #  f(x_i(t+1)) >=  f(y_i(t))
            if f_x_i >= s_f[i]:
                # y_i(t+1) = y_i
                # i.e. do nothing
                pass
            else:
                # new personal best position
                s_f[i] = f_x_i
                y_f[i] = s[i]

        f_min_ind = s_f.index(min(s_f))
        y_hat_f = y_f[f_min_ind]  # gets the first y_f if same fitness value
        L_f = L[f_min_ind]
        U_hat_f = U_hat[f_min_ind]
        w_f = w[f_min_ind]

        r_1: np.ndarray = np.random.rand(N)
        r_2: np.ndarray = np.random.rand(N)

        for i in range(1, P+1):
            for j in range(1, N+1):
                # update the velocity vector
                v[i][j] = round(v[i][j] + c_1*r_1[j]*(y_f[i][j]-s[i][j]) + c_2*r_2[j]*(y_hat_f[j]-s[i][j]))

                # perform velocity clamping
                v[i][j] = v[i][j] if abs(v[i][j]) < V_max else (abs(v[i][j])/v[i][j])*V_max
            # update particle position
            s[i] = s[i] + v[i]

        if t == 1: R_max_1 = np.max(np.linalg.norm(s-y_hat_f, axis=1))
        R_max_t = np.max(np.linalg.norm(s-y_hat_f, axis=1))

        t += 1

        end_loop = t > max_simulation_step and R_norm_threshold > R_max_t / R_max_1

    return L_f, U_hat_f, w_f


if __name__ == '__main__':
    pass

