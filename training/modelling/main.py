import random
import numpy as np
import pandas as pd

"""
https://ieeexplore.ieee.org/document/6395832
"""


def subspace_input_distance(u_k, *, l, N: int, a: list[list[int]], delta_a: list[int]):  # d^l_r
    return sum([
        (a[l-1][i-1]-u_k[i-1])**2/(N*delta_a[i-1]**2) for i in range(1, N+1)
    ])**.5

def nsfm_algorithm(*, U_train, Y_train, s: list[int]) -> (int, list[list[int]]):
    """
    :param U_train:
    :param Y_train:
    :param s:
    :return:
        L: number of selected RBF centers
        U: selected RBF center locations
    """
    N = len(s)
    L = 1  # prepare calculating the first RBF center (0-INDEXED)

    # s[i] fuzzy set partitioning of dim. i
    delta_a = [(max(U_train["col/dim i"])-min(U_train["col/dim i"])) / (s[i-1]-1) for i in range(1, N+1)]
    # fuzzy set centers (s[i]+1 for inclusive range)
    a = [[min(U_train["col/dim i"])+delta_a[i-1]*j for j in range(0, s[i-1]+1)] for i in range(1, N+1)]

    u_hat_1 = [] # generate the first RBF centers
    for i in range(1, N+1):  # 1 to N (1-indexed)
        # A^L_i, the center (partitioned by 1<=j<=s_i) with the max. membership for that dim. i & L
        # A_i,j to be the partitioning of each dim. i
        # we're using max instead of if/else <= 1, since d is always positive (refer to equation 9)
        # the equation is derived to be simpler by assuming the current dim. i
        # iterating each points of the partition (j over range(1,s_i))
        A_L_i = max([
            min(1-((a[i-1][j-1]-U_train["col/dim i"]["first row data"])**2/(N*delta_a[i-1])**2)**0.5, 1)
            for j in range(1, s[i-1]+1)
        ])
        u_hat_1.append(A_L_i)
    u_hat = [u_hat_1]

    k: int  # data point
    for k in range(1, len(U_train)+1):  # for each data point (k=1 has been done by initialization on previous loop)
        # if data point k lies outside the hyper-ellipses defined by the already selected center
        if min([subspace_input_distance(U_train["kth row data"], l=l, N=N, a=u_hat, delta_a=delta_a) for l in range(1, L+1)]) > 1:
            L += 1

            u_hat_L = []  # generate the first RBF centers
            for i in range(1, N + 1):  # 1 to N (1-indexed)
                A_L_i = max([
                    min(1-((a[i-1][j-1]-U_train["col/dim i"]["first row data"])**2/(N*delta_a[i-1])**2)**0.5, 1)
                    for j in range(1, s[i - 1] + 1)
                ])
                u_hat_L.append(A_L_i)
            u_hat.append(u_hat_L)  # generate Lth RBF centers

    return L, u_hat


def rbf_network(u_k, *, u_hat_l, N):
    mu_l = lambda u_k, u_hat, l: (sum([u_k[i]-u_hat[l][i] for i in range(1, N+1)]))**0.5

    Z = 0
    return Z

def pso_algorithm(U_train, Y_train: np.ndarray, U_val, Y_val, *, s_min: int, s_max: int, P: int, c_1: int, c_2: int, V_max: int) -> (int, list[int], list[int]):
    """
    :param U_train:
    :param Y_train:
    :param U_val:
    :param Y_val:
    :param s_min:
    :param s_max:
    :param P:
    :param c_1:
    :param c_2:
    :param V_max:
    :return:
        L_f: number of selected RBF centers
        U_f: selected RBF center locations
        w_f: selected RBF synaptic weights (for the last linear layer)
    """
    N = 5  # feature dimension
    s = [[random.randint(s_min, s_max) for _ in range(1, N+1)] for _i in range(1, P+1)]  # populations of N-lengthed fuzzy set
    t = 1

    L = []
    U = []
    w = []

    while True:
        # create the RBF-NN network
        for i in range(1, P+1):
            L_i, U_i = nsfm_algorithm(U_train=U_train, Y_train=Y_train, s=s[i])

            # calculate the synaptic weights
            np.dot(np.transpose(Y_train), Z)

            # calculate fitness function and update personal/global best positions if needed

        for i in range(1, P+1):
            for j in range(1, N+1):
                # update the velocity vector
                # perform velocity clamping
                pass
            # update particle position
        t += 1
