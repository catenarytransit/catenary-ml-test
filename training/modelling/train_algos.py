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

def nsfm_algorithm(*, U_train, Y_train, s: np.ndarray) -> (int, list[list[int]]):
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

    max_U_train = np.max(U_train.to_numpy(), axis=0)
    min_U_train = np.min(U_train.to_numpy(), axis=0)

    # s[i] fuzzy set partitioning of dim. i
    delta_a = [(max_U_train[i-1]-min_U_train[i-1]) / (s[i-1]-1) for i in range(1, N+1)]
    # fuzzy set centers (s[i]+1 for inclusive range)
    a = [[min_U_train[i-1]+delta_a[i-1]*j for j in range(0, s[i-1]+1)] for i in range(1, N+1)]

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
        u_hat_1.append(A_L_i)
    u_hat = [u_hat_1]

    k: int  # data point
    for k in range(2, len(U_train)+1):  # for each data point (k=1 has been done by initialization on previous loop)
        # if data point k lies outside the hyper-ellipses defined by the already selected center
        if min([subspace_input_distance(U_train[k-1], l=l, N=N, a=u_hat, delta_a=delta_a) for l in range(1, L+1)]) > 1:
            L += 1

            u_hat_L = []  # generate the first RBF centers
            for i in range(1, N+1):  # 1 to N (1-indexed)
                A_L_i = max([
                    min(1-((a[i-1][j-1]-U_train[1-1][i-1])**2/(N*delta_a[i-1])**2)**0.5, 1)
                    for j in range(1, s[i-1]+1)
                ])
                u_hat_L.append(A_L_i)
            u_hat.append(u_hat_L)  # generate Lth RBF centers

    return L, u_hat


class RBFNetwork:
    def __init__(self, *, u_hat: np.ndarray, L: int, N: int, ):
        self.N = N
        self.L = L

        self.w: np.ndarray = np.ones((L,))
        self.u_hat: np.ndarray = u_hat

        self.mu_l = lambda u_k, u_hat_l: np.linalg.norm(u_k-u_hat_l)

        # g(mu)
        self.activation = lambda mu, l: (mu**2)*np.log(mu)


    def forward(self, u_k) -> np.ndarray:
        # hidden node response
        self.z_k = [self.activation(self.mu_l(u_k, self.u_hat[l]), l) for l in range(1, self.L+1)]
        return np.dot(np.ndarray(self.z_k), self.w)

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
                  s_min: int, s_max: int, P: int, c_1: int, c_2: int, V_max: int,
                  max_simulation_step=100, R_norm_threshold=1) -> (int, list[int], list[int]):
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
    L = []
    U = []
    w = []

    R_max_1 = None
    R_max_t = None

    while t < max_simulation_step and \
        ((R_max_1 is not None and R_max_t is not None) and R_norm_threshold <= R_max_t / R_max_1):
        # create the RBF-NN network
        for i in range(1, P+1):
            L_i, u_hat_i = nsfm_algorithm(U_train=U_train, Y_train=Y_train, s=s[i])

            # calculate the synaptic weights
            rbf = RBFNetwork(u_hat=u_hat_i, L=L_i, N=N)
            rbf.train(U_train=U_train, Y_train=Y_train)
            w_i = rbf.w

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

        y_hat_f = y_f[s_f.index(min(s_f))]  # gets the first y_f if same fitness value

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


if __name__ == '__main__':
    pass

